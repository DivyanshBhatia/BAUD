#!/usr/bin/env python3
"""
BAUD: Run full pipeline on synthetic data.

This script:
1. Generates synthetic patients (stoic, expressive, moderate, random)
2. Runs BAUD calibration + pain detection on each
3. Runs all baselines for comparison
4. Generates all visualizations and metrics
5. Saves everything to results/

Usage:
    python scripts/run_synthetic.py
"""
import sys
import os
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    RESULTS_DIR, RANDOM_SEED, CALIBRATION_DURATIONS,
    SYNTHETIC_BASELINE_FRAMES, SYNTHETIC_PAIN_FRAMES,
    NUM_SYNTHETIC_PATIENTS,
)
from data.synthetic import generate_patient, generate_cohort, split_cohort
from models.baud import BAUDCalibrator
from baselines.methods import (
    GenericPainDetector, PSPICalculator, OneClassSVMBaseline,
    IsolationForestBaseline, MahalanobisBaseline,
)
from utils.metrics import (
    binary_pain_metrics, multilevel_pain_metrics,
    find_optimal_threshold, format_metrics_table,
)
from utils.visualization import (
    plot_pain_scores_comparison, plot_au_deviation_heatmap,
    plot_calibration_ablation, plot_personalization_comparison,
    generate_clinical_report,
)


def run_single_patient_experiment(patient_data, patient_name):
    """Run BAUD + all baselines on a single patient."""
    baseline_aus = patient_data["baseline_aus"]
    pain_aus = patient_data["pain_aus"]
    labels = patient_data["pain_labels"]

    # --- BAUD (ours) ---
    baud = BAUDCalibrator()
    baud.calibrate(baseline_aus)
    baud_scores, z_scores, reports = baud.process_sequence(pain_aus)

    # --- Baselines ---
    generic = GenericPainDetector()
    generic_scores = generic.process_sequence(pain_aus)

    pspi = PSPICalculator()
    pspi_scores = pspi.process_sequence(pain_aus)

    ocsvm = OneClassSVMBaseline()
    ocsvm.calibrate(baseline_aus)
    ocsvm_scores = ocsvm.process_sequence(pain_aus)

    iforest = IsolationForestBaseline()
    iforest.calibrate(baseline_aus)
    iforest_scores = iforest.process_sequence(pain_aus)

    mahal = MahalanobisBaseline()
    mahal.calibrate(baseline_aus)
    mahal_scores = mahal.process_sequence(pain_aus)

    # --- Metrics ---
    all_scores = {
        "BAUD (Ours)": baud_scores,
        "Generic (no personal.)": generic_scores,
        "PSPI Formula": pspi_scores,
        "One-Class SVM": ocsvm_scores,
        "Isolation Forest": iforest_scores,
        "Mahalanobis Distance": mahal_scores,
    }

    results = {}
    for name, scores in all_scores.items():
        # Find optimal threshold for fair comparison
        thresh = find_optimal_threshold(scores, labels)
        metrics = binary_pain_metrics(scores, labels, threshold=thresh)
        multi = multilevel_pain_metrics(scores, labels)
        metrics.update(multi)
        metrics["threshold"] = thresh
        results[name] = metrics

    return {
        "scores": all_scores,
        "metrics": results,
        "z_scores": z_scores,
        "reports": reports,
        "labels": labels,
        "baseline_aus": baseline_aus,
        "pain_aus": pain_aus,
        "patient_name": patient_name,
    }


def run_calibration_ablation(patient_data):
    """Test how calibration duration affects performance."""
    pain_aus = patient_data["pain_aus"]
    labels = patient_data["pain_labels"]
    full_baseline = patient_data["baseline_aus"]

    f1_scores = []
    auc_scores = []
    durations = CALIBRATION_DURATIONS

    for n_frames in durations:
        # Use first n_frames of baseline for calibration
        cal_frames = min(n_frames, len(full_baseline))
        subset = full_baseline[:cal_frames]

        baud = BAUDCalibrator()
        baud.calibrate(subset)
        scores, _, _ = baud.process_sequence(pain_aus)

        thresh = find_optimal_threshold(scores, labels)
        metrics = binary_pain_metrics(scores, labels, threshold=thresh)
        f1_scores.append(metrics["f1"])
        auc_scores.append(metrics["auc"])

    return durations, f1_scores, auc_scores


def run_cohort_experiment():
    """Run BAUD across a cohort and aggregate metrics."""
    print("\n" + "=" * 60)
    print("  Running cohort-level experiment")
    print("=" * 60)

    cohort = generate_cohort(
        num_patients=NUM_SYNTHETIC_PATIENTS,
        num_baseline=SYNTHETIC_BASELINE_FRAMES,
        num_pain=SYNTHETIC_PAIN_FRAMES,
        seed=RANDOM_SEED,
    )
    train, val, test = split_cohort(cohort, seed=RANDOM_SEED)

    # Aggregate metrics across test patients
    method_names = [
        "BAUD (Ours)", "Generic (no personal.)", "PSPI Formula",
        "One-Class SVM", "Isolation Forest", "Mahalanobis Distance",
    ]
    aggregated = {name: [] for name in method_names}

    for patient in test:
        pid = patient["patient_id"]
        ptype = patient["metadata"]["patient_type"]
        result = run_single_patient_experiment(
            patient, f"Patient {pid} ({ptype})"
        )
        for name in method_names:
            if name in result["metrics"]:
                aggregated[name].append(result["metrics"][name])

    # Average metrics
    avg_results = {}
    for name, metric_list in aggregated.items():
        if metric_list:
            avg_results[name] = {
                k: float(np.mean([m[k] for m in metric_list]))
                for k in metric_list[0].keys()
            }

    return avg_results


def main():
    print("=" * 60)
    print("  BAUD: Baseline-Anchored AU Deviation")
    print("  Running Synthetic Data Experiments")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ================================================================
    # Experiment 1: Two contrasting patients (stoic vs expressive)
    # ================================================================
    print("\n📊 Experiment 1: Stoic vs Expressive Patient Comparison")
    print("-" * 60)

    patient_A = generate_patient("stoic", seed=42)
    patient_B = generate_patient("expressive", seed=99)

    result_A = run_single_patient_experiment(patient_A, "Patient A (Stoic)")
    result_B = run_single_patient_experiment(patient_B, "Patient B (Expressive)")

    # Print metrics
    print("\n" + format_metrics_table(
        result_A["metrics"], "Patient A (Stoic) — Binary Pain Detection"
    ))
    print("\n" + format_metrics_table(
        result_B["metrics"], "Patient B (Expressive) — Binary Pain Detection"
    ))

    # Save plots
    print("\n📈 Generating visualizations...")

    plot_pain_scores_comparison(
        result_A["scores"], result_A["labels"], "Patient A (Stoic)",
        save_path=os.path.join(RESULTS_DIR, "pain_scores_patient_A.png"),
    )
    plot_pain_scores_comparison(
        result_B["scores"], result_B["labels"], "Patient B (Expressive)",
        save_path=os.path.join(RESULTS_DIR, "pain_scores_patient_B.png"),
    )

    plot_au_deviation_heatmap(
        {
            "Patient A (Stoic)": result_A["z_scores"],
            "Patient B (Expressive)": result_B["z_scores"],
        },
        save_path=os.path.join(RESULTS_DIR, "au_deviation_heatmap.png"),
    )

    plot_personalization_comparison(
        result_A["pain_aus"], result_B["pain_aus"],
        result_A["z_scores"], result_B["z_scores"],
        frame_idx=70,
        save_path=os.path.join(RESULTS_DIR, "personalization_comparison.png"),
    )

    # ================================================================
    # Experiment 2: Calibration duration ablation
    # ================================================================
    print("\n📊 Experiment 2: Calibration Duration Ablation")
    print("-" * 60)

    durations, f1s, aucs = run_calibration_ablation(patient_A)
    print(f"  {'Frames':<10} {'F1':>8} {'AUC':>8}")
    print("  " + "-" * 30)
    for d, f, a in zip(durations, f1s, aucs):
        print(f"  {d:<10} {f:>8.4f} {a:>8.4f}")

    plot_calibration_ablation(
        durations, f1s, aucs,
        save_path=os.path.join(RESULTS_DIR, "calibration_ablation.png"),
    )

    # ================================================================
    # Experiment 3: Cohort-level evaluation
    # ================================================================
    print("\n📊 Experiment 3: Cohort-Level Evaluation")
    print("-" * 60)

    avg_results = run_cohort_experiment()
    cohort_table = format_metrics_table(
        avg_results, "Cohort Average — Binary Pain Detection"
    )
    print("\n" + cohort_table)

    # ================================================================
    # Clinical reports
    # ================================================================
    print("\n📋 Generating Clinical Reports...")
    print("-" * 60)

    report_A = generate_clinical_report(
        result_A["reports"], result_A["scores"]["BAUD (Ours)"],
        result_A["labels"], "Patient A (Stoic)"
    )
    report_B = generate_clinical_report(
        result_B["reports"], result_B["scores"]["BAUD (Ours)"],
        result_B["labels"], "Patient B (Expressive)"
    )

    print(report_A)
    print(report_B)

    # ================================================================
    # Save all text results
    # ================================================================
    results_file = os.path.join(RESULTS_DIR, "metrics_table.txt")
    with open(results_file, "w") as f:
        f.write("BAUD: Baseline-Anchored AU Deviation — Results\n")
        f.write("=" * 75 + "\n\n")
        f.write(format_metrics_table(
            result_A["metrics"], "Patient A (Stoic)") + "\n\n")
        f.write(format_metrics_table(
            result_B["metrics"], "Patient B (Expressive)") + "\n\n")
        f.write(cohort_table + "\n\n")
        f.write("Calibration Duration Ablation:\n")
        f.write(f"{'Frames':<10} {'F1':>8} {'AUC':>8}\n")
        for d, f1, auc in zip(durations, f1s, aucs):
            f.write(f"{d:<10} {f1:>8.4f} {auc:>8.4f}\n")
    print(f"\n  Saved: {results_file}")

    clinical_file = os.path.join(RESULTS_DIR, "clinical_report.txt")
    with open(clinical_file, "w") as f:
        f.write(report_A + "\n\n" + report_B)
    print(f"  Saved: {clinical_file}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("  ✅ ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\n  Output files in: {RESULTS_DIR}/")
    print("  ├── pain_scores_patient_A.png")
    print("  ├── pain_scores_patient_B.png")
    print("  ├── au_deviation_heatmap.png")
    print("  ├── personalization_comparison.png")
    print("  ├── calibration_ablation.png")
    print("  ├── metrics_table.txt")
    print("  └── clinical_report.txt")
    print("\n  Share these files to review results!")
    print("=" * 60)


if __name__ == "__main__":
    main()
