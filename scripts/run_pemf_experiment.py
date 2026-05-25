#!/usr/bin/env python3
"""
BAUD × PEMF: Run personalized pain detection on real PEMF data.

Run this on Google Colab with PEMF dataset at /content/pemf/

Usage:
    python run_pemf_experiment.py
"""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
PEMF_ROOT = "/content/pemf"
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pain-related AU columns in PEMF xlsx (FACS coded, 0-6 intensity scale)
PAIN_AU_COLS = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
ALL_AU_COLS = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU12",
               "AU20", "AU25", "AU26", "AU27", "AU43", "AU45"]
NUM_AUS = len(ALL_AU_COLS)

EXPR_MAP = {
    "Neutral": {"code": "N", "pain_level": 0},
    "Algometer": {"code": "A", "pain_level": 2},
    "Laser": {"code": "L", "pain_level": 1},
    "Posed": {"code": "P", "pain_level": 3},
}

SEED = 42

# ============================================================================
# STEP 1: Load PEMF metadata and AU annotations
# ============================================================================

def load_pemf_metadata(root_dir):
    """Load PEMF_Database.xlsx and extract per-subject AU data."""
    xlsx_path = os.path.join(root_dir, "PEMF_Database.xlsx")
    df = pd.read_excel(xlsx_path)

    subjects = {}
    for _, row in df.iterrows():
        clip = str(row["Clip"]).strip()
        subj_id = clip[:-1]   # e.g., "S001" from "S001A"
        expr_code = clip[-1]  # e.g., "A", "L", "N", "P"

        # Parse AU values (FACS intensity 0-6)
        au_vector = []
        for au_col in ALL_AU_COLS:
            val = row.get(au_col, 0)
            try:
                au_vector.append(float(val) if pd.notna(val) else 0.0)
            except (ValueError, TypeError):
                au_vector.append(0.0)
        au_vector = np.array(au_vector)

        # Normalize to [0, 1] (FACS uses 0-6 scale)
        au_vector = au_vector / 6.0

        # Parse intensity rating
        intensity_str = str(row.get("Intensity", "0"))
        try:
            # Handle European decimal format: "5,20 (2,516)" → 5.20
            intensity = float(intensity_str.split("(")[0].strip().replace(",", "."))
        except (ValueError, AttributeError):
            intensity = 0.0

        if subj_id not in subjects:
            subjects[subj_id] = {
                "age": row.get("Age", None),
                "gender": row.get("Gender", None),
                "expressions": {},
            }

        # Map code to expression name
        code_to_expr = {"N": "Neutral", "A": "Algometer",
                        "L": "Laser", "P": "Posed"}
        expr_name = code_to_expr.get(expr_code, expr_code)

        subjects[subj_id]["expressions"][expr_name] = {
            "au_vector": au_vector,
            "intensity": intensity,
            "clip_code": clip,
        }

    return subjects


# ============================================================================
# STEP 2: BAUD Calibrator (adapted for PEMF)
# ============================================================================

class BAUDCalibrator:
    """BAUD calibrator using FACS AU annotations from PEMF."""

    def __init__(self, num_aus=NUM_AUS):
        self.num_aus = num_aus
        self.baseline_au = None
        self.is_calibrated = False

        # Pain-related AU indices within ALL_AU_COLS
        self.pain_indices = [ALL_AU_COLS.index(au) for au in PAIN_AU_COLS
                             if au in ALL_AU_COLS]

        # Prior deviation weights
        self.weights = np.ones(num_aus)
        for idx in self.pain_indices:
            self.weights[idx] = 3.0
        self.weights /= self.weights.sum()

    def calibrate(self, neutral_au_vector):
        """Calibrate using the neutral (baseline) AU vector."""
        self.baseline_au = neutral_au_vector.copy()
        # For single-frame calibration, use a small assumed std
        # based on typical AU variation
        self.baseline_std = np.maximum(np.abs(neutral_au_vector) * 0.3 + 0.02, 0.02)
        self.is_calibrated = True

    def score(self, au_vector):
        """Score a pain AU vector against the baseline."""
        assert self.is_calibrated
        # Per-AU deviation
        deviation = au_vector - self.baseline_au
        z_scores = deviation / self.baseline_std
        z_positive = np.maximum(z_scores, 0)

        # Weighted score
        raw = np.dot(self.weights, z_positive)
        pain_score = 1.0 / (1.0 + np.exp(-raw + 2.0))

        return {
            "pain_score": float(pain_score),
            "z_scores": z_scores,
            "deviation": deviation,
        }


# ============================================================================
# STEP 3: Baseline methods
# ============================================================================

def generic_pain_score(au_vector, pain_indices):
    """Non-personalized: average of pain-related AU values."""
    return float(np.mean(au_vector[pain_indices]))


def pspi_score(au_vector):
    """PSPI formula: AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43"""
    au4 = au_vector[ALL_AU_COLS.index("AU4")]
    au6 = au_vector[ALL_AU_COLS.index("AU6")]
    au7 = au_vector[ALL_AU_COLS.index("AU7")]
    au9 = au_vector[ALL_AU_COLS.index("AU9")]
    au10 = au_vector[ALL_AU_COLS.index("AU10")]
    au43 = au_vector[ALL_AU_COLS.index("AU43")]
    pspi = au4 + max(au6, au7) + max(au9, au10) + au43
    return float(min(pspi / 4.0, 1.0))  # Normalize to [0,1]


# ============================================================================
# STEP 4: Run experiments
# ============================================================================

def run_experiment(subjects, test_subjects):
    """Run BAUD + baselines on test subjects."""
    results = defaultdict(list)
    per_subject_results = []

    for subj_id in test_subjects:
        subj = subjects[subj_id]
        exprs = subj["expressions"]

        if "Neutral" not in exprs:
            continue

        neutral_au = exprs["Neutral"]["au_vector"]

        # Calibrate BAUD
        baud = BAUDCalibrator()
        baud.calibrate(neutral_au)

        for expr_name in ["Algometer", "Laser", "Posed"]:
            if expr_name not in exprs:
                continue

            pain_au = exprs[expr_name]["au_vector"]
            intensity = exprs[expr_name]["intensity"]
            is_pain = 1  # All non-neutral are pain

            # BAUD score
            baud_result = baud.score(pain_au)
            baud_score = baud_result["pain_score"]

            # Generic score
            gen_score = generic_pain_score(pain_au, baud.pain_indices)

            # PSPI score
            pspi = pspi_score(pain_au)

            results["BAUD (Ours)"].append({
                "score": baud_score, "true": is_pain,
                "intensity": intensity, "subject": subj_id, "expr": expr_name,
            })
            results["Generic"].append({
                "score": gen_score, "true": is_pain,
                "intensity": intensity, "subject": subj_id, "expr": expr_name,
            })
            results["PSPI"].append({
                "score": pspi, "true": is_pain,
                "intensity": intensity, "subject": subj_id, "expr": expr_name,
            })

            per_subject_results.append({
                "subject": subj_id,
                "expression": expr_name,
                "age": subj["age"],
                "gender": subj["gender"],
                "intensity_rating": intensity,
                "baud_score": baud_score,
                "generic_score": gen_score,
                "pspi_score": pspi,
                "z_scores": baud_result["z_scores"],
            })

        # Also score neutral against itself (should be low pain)
        baud_neutral = baud.score(neutral_au)
        for method, score in [("BAUD (Ours)", baud_neutral["pain_score"]),
                               ("Generic", generic_pain_score(neutral_au, baud.pain_indices)),
                               ("PSPI", pspi_score(neutral_au))]:
            results[method].append({
                "score": score, "true": 0, "intensity": 0,
                "subject": subj_id, "expr": "Neutral",
            })

    return results, per_subject_results


def compute_metrics(results):
    """Compute binary classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics = {}
    for method_name, entries in results.items():
        scores = [e["score"] for e in entries]
        truths = [e["true"] for e in entries]

        # Find best threshold
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            preds = [1 if s > t else 0 for s in scores]
            f1 = f1_score(truths, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        preds = [1 if s > best_thresh else 0 for s in scores]
        try:
            auc = roc_auc_score(truths, scores)
        except ValueError:
            auc = 0.0

        metrics[method_name] = {
            "accuracy": accuracy_score(truths, preds),
            "f1": best_f1,
            "auc": auc,
            "threshold": best_thresh,
            "n_samples": len(entries),
        }
    return metrics


# ============================================================================
# STEP 5: Visualizations
# ============================================================================

def plot_score_distribution(results, save_path):
    """Plot pain score distributions for pain vs neutral."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"BAUD (Ours)": "#2196F3", "Generic": "#FF5722", "PSPI": "#4CAF50"}

    for ax, (method, entries) in zip(axes, results.items()):
        pain_scores = [e["score"] for e in entries if e["true"] == 1]
        neutral_scores = [e["score"] for e in entries if e["true"] == 0]

        ax.hist(neutral_scores, bins=15, alpha=0.7, label="Neutral",
                color="#90CAF9", edgecolor="white")
        ax.hist(pain_scores, bins=15, alpha=0.7, label="Pain",
                color="#EF5350", edgecolor="white")
        ax.set_title(method, fontsize=13, fontweight="bold")
        ax.set_xlabel("Pain Score")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Pain Score Distribution: Neutral vs Pain",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_subject_comparison(per_subject_results, save_path):
    """Scatter plot: BAUD vs Generic scores colored by pain intensity."""
    fig, ax = plt.subplots(figsize=(8, 8))

    baud_scores = [r["baud_score"] for r in per_subject_results]
    generic_scores = [r["generic_score"] for r in per_subject_results]
    intensities = [r["intensity_rating"] for r in per_subject_results]

    scatter = ax.scatter(generic_scores, baud_scores, c=intensities,
                         cmap="YlOrRd", s=80, alpha=0.7, edgecolors="white")
    plt.colorbar(scatter, ax=ax, label="Pain Intensity Rating")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Generic Score (no personalization)", fontsize=12)
    ax.set_ylabel("BAUD Score (personalized)", fontsize=12)
    ax.set_title("BAUD vs Generic: Per-Subject Pain Scores",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_au_deviation_per_expression(per_subject_results, save_path):
    """Bar chart: average AU z-scores per expression type."""
    expr_z_scores = defaultdict(list)
    for r in per_subject_results:
        expr_z_scores[r["expression"]].append(r["z_scores"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pain_au_labels = PAIN_AU_COLS

    for ax, expr_name in zip(axes, ["Algometer", "Laser", "Posed"]):
        if expr_name not in expr_z_scores:
            continue

        z_matrix = np.array(expr_z_scores[expr_name])
        pain_indices = [ALL_AU_COLS.index(au) for au in PAIN_AU_COLS]
        mean_z = np.mean(z_matrix[:, pain_indices], axis=0)
        std_z = np.std(z_matrix[:, pain_indices], axis=0)

        colors = ["#EF5350" if z > 1.5 else "#FFA726" if z > 0.5
                  else "#66BB6A" for z in mean_z]
        bars = ax.bar(range(len(pain_au_labels)), np.maximum(mean_z, 0),
                      yerr=std_z, color=colors, edgecolor="white",
                      capsize=3, alpha=0.8)
        ax.set_xticks(range(len(pain_au_labels)))
        ax.set_xticklabels(pain_au_labels, fontsize=10)
        ax.set_ylabel("Z-Score (σ from baseline)")
        ax.set_title(f"{expr_name} Pain", fontsize=13, fontweight="bold")
        ax.axhline(y=1.5, color="red", linestyle=":", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("BAUD: Per-AU Deviation from Baseline by Pain Type",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_report(per_subject_results, subj_id, save_path=None):
    """Generate clinical-style report for one subject."""
    subj_results = [r for r in per_subject_results if r["subject"] == subj_id]
    if not subj_results:
        return ""

    lines = []
    lines.append("=" * 60)
    lines.append(f"  BAUD Clinical Report — {subj_id}")
    lines.append(f"  Age: {subj_results[0]['age']}, "
                 f"Gender: {subj_results[0]['gender']}")
    lines.append("=" * 60)

    for r in subj_results:
        lines.append(f"\n  Expression: {r['expression']}")
        lines.append(f"  Pain Intensity Rating: {r['intensity_rating']:.1f}")
        lines.append(f"  BAUD Score: {r['baud_score']:.3f}")
        lines.append(f"  Generic Score: {r['generic_score']:.3f}")
        lines.append(f"  PSPI Score: {r['pspi_score']:.3f}")
        lines.append("  " + "-" * 40)
        lines.append("  Per-AU Deviations from Baseline:")

        pain_indices = [ALL_AU_COLS.index(au) for au in PAIN_AU_COLS]
        for au_name, idx in zip(PAIN_AU_COLS, pain_indices):
            z = r["z_scores"][idx]
            bar_len = int(min(abs(z), 10))
            bar = "█" * bar_len
            flag = " ⚠️" if z > 1.5 else ""
            lines.append(f"    {au_name:>5s}: {z:>+6.2f}σ  {bar:<10s}{flag}")

    lines.append("\n" + "=" * 60)
    report = "\n".join(lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"  Saved: {save_path}")

    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  BAUD × PEMF: Real Data Experiment")
    print("=" * 60)

    # Load PEMF metadata
    print("\n📂 Loading PEMF metadata...")
    subjects = load_pemf_metadata(PEMF_ROOT)
    print(f"  Loaded {len(subjects)} subjects")

    # Split subjects
    all_subj_ids = sorted(subjects.keys())
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(all_subj_ids))

    n_train = int(len(all_subj_ids) * 0.7)
    n_val = int(len(all_subj_ids) * 0.1)

    train_ids = [all_subj_ids[i] for i in indices[:n_train]]
    val_ids = [all_subj_ids[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [all_subj_ids[i] for i in indices[n_train + n_val:]]

    print(f"  Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"  Test subjects: {test_ids}")

    # Run experiments on test subjects
    print("\n📊 Running experiments...")
    results, per_subject_results = run_experiment(subjects, test_ids)

    # Compute metrics
    print("\n📈 Computing metrics...")
    metrics = compute_metrics(results)

    # Print metrics table
    print("\n" + "=" * 70)
    print(f"  {'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Thresh':>8}")
    print("=" * 70)
    for method, m in metrics.items():
        print(f"  {method:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {m['threshold']:>8.2f}")
    print("=" * 70)

    # Also run on ALL subjects for more robust numbers
    print("\n📊 Running on ALL subjects...")
    all_results, all_per_subject = run_experiment(subjects, all_subj_ids)
    all_metrics = compute_metrics(all_results)

    print("\n" + "=" * 70)
    print(f"  ALL SUBJECTS — {'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("=" * 70)
    for method, m in all_metrics.items():
        print(f"  {method:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f}")
    print("=" * 70)

    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_score_distribution(
        all_results, os.path.join(RESULTS_DIR, "pemf_score_distribution.png")
    )
    plot_per_subject_comparison(
        all_per_subject, os.path.join(RESULTS_DIR, "pemf_baud_vs_generic.png")
    )
    plot_au_deviation_per_expression(
        all_per_subject, os.path.join(RESULTS_DIR, "pemf_au_deviations.png")
    )

    # Clinical reports for a few test subjects
    print("\n📋 Generating clinical reports...")
    for subj_id in test_ids[:3]:
        report = generate_report(
            all_per_subject, subj_id,
            save_path=os.path.join(RESULTS_DIR, f"report_{subj_id}.txt")
        )
        print(report)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "pemf_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("BAUD × PEMF Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}\n")
        f.write("-" * 70 + "\n")
        for method, m in all_metrics.items():
            f.write(f"{method:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
                    f"{m['auc']:>8.4f}\n")
    print(f"  Saved: {metrics_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ PEMF EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\n  Results in: {RESULTS_DIR}/")
    print("  ├── pemf_score_distribution.png")
    print("  ├── pemf_baud_vs_generic.png")
    print("  ├── pemf_au_deviations.png")
    print("  ├── pemf_metrics.txt")
    print("  └── report_S0XX.txt (clinical reports)")
    print("\n  Share these files to review results!")


if __name__ == "__main__":
    main()
