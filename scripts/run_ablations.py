#!/usr/bin/env python3
"""
BAUD Ablation Experiments on PEMF
==================================

Experiment 1: Calibration Duration — How many baseline frames do you need?
Experiment 2: Per-Expression Analysis — Which pain types are hardest?
Experiment 3: Leave-One-Expression-Out — Train on 2 pain types, test on 3rd
Experiment 4: Per-Subject Analysis — Which patients benefit most from personalization?
Experiment 5: AU Encoder Ablation — BAUD on subsets of AUs

Usage:
    python scripts/run_ablations.py
"""
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import OneClassSVM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "/content/results"
CACHE_PATH = "/content/results/pemf_extracted_aus.npz"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_AUS = 41
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
SEED = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    cached = np.load(CACHE_PATH)
    au_data = {}
    for key in cached.files:
        subj, expr = key[:4], key[5:]
        if subj not in au_data:
            au_data[subj] = {}
        au_data[subj][expr] = cached[key]
    return au_data


# ============================================================================
# SCORING METHODS
# ============================================================================

def baud_score(neutral_aus, test_aus, n_cal_frames=None):
    """BAUD with optional frame limit for calibration."""
    if n_cal_frames is not None:
        neutral_aus = neutral_aus[:n_cal_frames]
    if len(neutral_aus) < 2:
        neutral_aus = np.vstack([neutral_aus, neutral_aus])  # duplicate if only 1

    mean_b = np.mean(neutral_aus, axis=0)
    std_b = np.maximum(np.std(neutral_aus, axis=0), 1e-4)

    w = np.ones(test_aus.shape[1])
    for idx in PAIN_IDX:
        if idx < len(w):
            w[idx] = 3.0
    w /= w.sum()

    z = np.maximum((test_aus - mean_b) / std_b, 0)
    raw = np.array([np.dot(w, f) for f in z])
    scores = 1.0 / (1.0 + np.exp(-raw + 2.0))
    return float(np.mean(scores))


def generic_score(test_aus):
    return float(np.mean([np.mean(f[PAIN_IDX]) for f in test_aus]))


def mahalanobis_score(neutral_aus, test_aus, n_cal_frames=None):
    if n_cal_frames is not None:
        neutral_aus = neutral_aus[:n_cal_frames]
    if len(neutral_aus) < 2:
        neutral_aus = np.vstack([neutral_aus, neutral_aus])

    feat_b = neutral_aus[:, PAIN_IDX]
    mean = np.mean(feat_b, 0)
    cov = np.cov(feat_b, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
    cov_inv = np.linalg.inv(cov)

    scores = []
    for f in test_aus[:, PAIN_IDX]:
        diff = f - mean
        d = float(np.sqrt(diff @ cov_inv @ diff))
        scores.append(1.0 / (1.0 + np.exp(-d + 3.0)))
    return float(np.mean(scores))


def compute_metrics(scores, truths):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    preds = [1 if s > best_t else 0 for s in scores]
    try:
        auc = roc_auc_score(truths, scores)
    except:
        auc = 0.0
    return {"acc": accuracy_score(truths, preds), "f1": best_f1, "auc": auc}


# ============================================================================
# EXPERIMENT 1: Calibration Duration Ablation
# ============================================================================

def exp1_calibration_duration(au_data):
    """How many neutral frames do you need for good calibration?"""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: Calibration Duration Ablation")
    print("  How many baseline frames does BAUD need?")
    print("=" * 65)

    durations = [1, 2, 3, 5, 8, 10, 15, 20]
    subjects = [s for s in au_data if "Neutral" in au_data[s]]

    results = {d: {"baud": [], "mahal": []} for d in durations}
    results["full"] = {"baud": [], "mahal": []}

    for subj in subjects:
        neutral = au_data[subj]["Neutral"]
        max_frames = len(neutral)

        for d in durations + ["full"]:
            n = max_frames if d == "full" else min(d, max_frames)

            # Score neutral (should be 0)
            ns_baud = baud_score(neutral, neutral, n)
            ns_mah = mahalanobis_score(neutral, neutral, n)

            results[d]["baud"].append({"score": ns_baud, "true": 0})
            results[d]["mahal"].append({"score": ns_mah, "true": 0})

            # Score pain
            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                pain = au_data[subj].get(expr)
                if pain is None:
                    continue
                ps_baud = baud_score(neutral, pain, n)
                ps_mah = mahalanobis_score(neutral, pain, n)

                results[d]["baud"].append({"score": ps_baud, "true": 1})
                results[d]["mahal"].append({"score": ps_mah, "true": 1})

    # Compute metrics per duration
    print(f"\n  {'Frames':<10} {'BAUD Acc':>10} {'BAUD F1':>10} {'BAUD AUC':>10} "
          f"{'Mah. AUC':>10}")
    print("  " + "-" * 55)

    baud_aucs, mah_aucs, baud_f1s = [], [], []

    for d in durations:
        bm = compute_metrics([e["score"] for e in results[d]["baud"]],
                             [e["true"] for e in results[d]["baud"]])
        mm = compute_metrics([e["score"] for e in results[d]["mahal"]],
                             [e["true"] for e in results[d]["mahal"]])
        baud_aucs.append(bm["auc"])
        baud_f1s.append(bm["f1"])
        mah_aucs.append(mm["auc"])
        print(f"  {d:<10} {bm['acc']:>10.4f} {bm['f1']:>10.4f} "
              f"{bm['auc']:>10.4f} {mm['auc']:>10.4f}")

    # Full baseline
    bm = compute_metrics([e["score"] for e in results["full"]["baud"]],
                         [e["true"] for e in results["full"]["baud"]])
    mm = compute_metrics([e["score"] for e in results["full"]["mahal"]],
                         [e["true"] for e in results["full"]["mahal"]])
    print(f"  {'All (20)':<10} {bm['acc']:>10.4f} {bm['f1']:>10.4f} "
          f"{bm['auc']:>10.4f} {mm['auc']:>10.4f}")

    return durations, baud_aucs, baud_f1s, mah_aucs


# ============================================================================
# EXPERIMENT 2: Per-Expression-Type Analysis
# ============================================================================

def exp2_per_expression(au_data):
    """How well does BAUD detect each pain type separately?"""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: Per-Expression-Type Analysis")
    print("  Which pain types are hardest to detect?")
    print("=" * 65)

    subjects = [s for s in au_data if "Neutral" in au_data[s]]
    expressions = ["Algometer Pain", "Laser Pain", "Posed Pain"]

    results = {}
    for expr in expressions:
        baud_scores, gen_scores, mah_scores, truths = [], [], [], []

        for subj in subjects:
            neutral = au_data[subj]["Neutral"]

            # Neutral vs this specific expression
            baud_scores.append(baud_score(neutral, neutral))
            gen_scores.append(generic_score(neutral))
            mah_scores.append(mahalanobis_score(neutral, neutral))
            truths.append(0)

            pain = au_data[subj].get(expr)
            if pain is None:
                continue
            baud_scores.append(baud_score(neutral, pain))
            gen_scores.append(generic_score(pain))
            mah_scores.append(mahalanobis_score(neutral, pain))
            truths.append(1)

        results[expr] = {
            "BAUD": compute_metrics(baud_scores, truths),
            "Generic": compute_metrics(gen_scores, truths),
            "Mahalanobis": compute_metrics(mah_scores, truths),
            "baud_pain_scores": [s for s, t in zip(baud_scores, truths) if t == 1],
            "baud_neutral_scores": [s for s, t in zip(baud_scores, truths) if t == 0],
            "gen_pain_scores": [s for s, t in zip(gen_scores, truths) if t == 1],
        }

    # Print table
    print(f"\n  {'Expression':<20} {'Method':<15} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("  " + "-" * 60)
    for expr in expressions:
        for method in ["BAUD", "Generic", "Mahalanobis"]:
            m = results[expr][method]
            prefix = "  " if method != "BAUD" else "  "
            print(f"{prefix}{expr:<20} {method:<15} {m['acc']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['auc']:>8.4f}")
        print("  " + "-" * 60)

    return results


# ============================================================================
# EXPERIMENT 3: Per-Subject Personalization Benefit
# ============================================================================

def exp3_per_subject_benefit(au_data):
    """For each subject, how much does personalization help?"""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: Per-Subject Personalization Benefit")
    print("  Which patients benefit most from BAUD?")
    print("=" * 65)

    subjects = sorted([s for s in au_data if "Neutral" in au_data[s]])
    per_subject = []

    for subj in subjects:
        neutral = au_data[subj]["Neutral"]
        baud_pain_scores = []
        gen_pain_scores = []
        baud_neutral = baud_score(neutral, neutral)
        gen_neutral = generic_score(neutral)

        for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            pain = au_data[subj].get(expr)
            if pain is None:
                continue
            baud_pain_scores.append(baud_score(neutral, pain))
            gen_pain_scores.append(generic_score(pain))

        if not baud_pain_scores:
            continue

        # Separation = mean_pain_score - neutral_score (higher = better separation)
        baud_sep = np.mean(baud_pain_scores) - baud_neutral
        gen_sep = np.mean(gen_pain_scores) - gen_neutral
        benefit = baud_sep - gen_sep

        # Baseline variability (std of neutral AUs) — indicates expressiveness
        baseline_var = float(np.mean(np.std(neutral, axis=0)))

        per_subject.append({
            "subject": subj,
            "baud_separation": baud_sep,
            "generic_separation": gen_sep,
            "personalization_benefit": benefit,
            "baseline_variability": baseline_var,
            "baud_neutral": baud_neutral,
            "baud_pain_mean": float(np.mean(baud_pain_scores)),
            "gen_neutral": gen_neutral,
            "gen_pain_mean": float(np.mean(gen_pain_scores)),
        })

    # Sort by personalization benefit
    per_subject.sort(key=lambda x: x["personalization_benefit"], reverse=True)

    print(f"\n  {'Subject':<10} {'BAUD Sep':>10} {'Gen Sep':>10} "
          f"{'Benefit':>10} {'Base Var':>10}")
    print("  " + "-" * 55)
    for p in per_subject[:10]:
        print(f"  {p['subject']:<10} {p['baud_separation']:>10.4f} "
              f"{p['generic_separation']:>10.4f} "
              f"{p['personalization_benefit']:>10.4f} "
              f"{p['baseline_variability']:>10.4f}")
    print("  ...")
    for p in per_subject[-5:]:
        print(f"  {p['subject']:<10} {p['baud_separation']:>10.4f} "
              f"{p['generic_separation']:>10.4f} "
              f"{p['personalization_benefit']:>10.4f} "
              f"{p['baseline_variability']:>10.4f}")

    return per_subject


# ============================================================================
# EXPERIMENT 4: AU Subset Ablation
# ============================================================================

def exp4_au_ablation(au_data):
    """How important are specific AU subsets?"""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4: AU Subset Ablation")
    print("  Which AU groups contribute most?")
    print("=" * 65)

    subjects = [s for s in au_data if "Neutral" in au_data[s]]

    configs = {
        "All 41 AUs": list(range(NUM_AUS)),
        "Pain AUs only (6)": PAIN_IDX,
        "Non-pain AUs only": [i for i in range(18) if i not in PAIN_IDX],
        "Upper face (AU1-7)": [0, 1, 2, 3, 4, 5],
        "Lower face (AU9-43)": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "Just AU4 + AU43": [2, 17],
        "Random 6 non-pain": [0, 1, 3, 8, 9, 10],
    }

    results = {}
    for config_name, au_indices in configs.items():
        scores, truths = [], []

        for subj in subjects:
            neutral = au_data[subj]["Neutral"][:, au_indices]
            mean_b = np.mean(neutral, 0)
            std_b = np.maximum(np.std(neutral, 0), 1e-4)
            w = np.ones(len(au_indices))
            w /= w.sum()

            # Neutral
            z = np.maximum((neutral - mean_b) / std_b, 0)
            raw = np.array([np.dot(w, f) for f in z])
            ns = float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))
            scores.append(ns)
            truths.append(0)

            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                pain_full = au_data[subj].get(expr)
                if pain_full is None:
                    continue
                pain = pain_full[:, au_indices]
                z = np.maximum((pain - mean_b) / std_b, 0)
                raw = np.array([np.dot(w, f) for f in z])
                ps = float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))
                scores.append(ps)
                truths.append(1)

        results[config_name] = compute_metrics(scores, truths)

    print(f"\n  {'AU Configuration':<30} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("  " + "-" * 55)
    for name, m in results.items():
        print(f"  {name:<30} {m['acc']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_all_ablations(dur_data, expr_data, subj_data, au_data, save_dir):
    """Generate all ablation plots."""

    # ── Figure 1: Calibration Duration ──
    durations, baud_aucs, baud_f1s, mah_aucs = dur_data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(durations, baud_f1s, "o-", color="#2196F3", linewidth=2,
             markersize=8, label="BAUD", zorder=5)
    ax1.axhline(y=baud_f1s[-1], color="#2196F3", linestyle=":", alpha=0.3)
    ax1.fill_between(durations, [f * 0.99 for f in baud_f1s],
                     [min(f * 1.01, 1.0) for f in baud_f1s],
                     alpha=0.1, color="#2196F3")
    ax1.set_xlabel("Number of Baseline Frames", fontsize=12)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("F1 vs Calibration Duration", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.85, 1.02)

    ax2.plot(durations, baud_aucs, "o-", color="#2196F3", linewidth=2,
             markersize=8, label="BAUD")
    ax2.plot(durations, mah_aucs, "s--", color="#FF9800", linewidth=2,
             markersize=8, label="Mahalanobis")
    ax2.set_xlabel("Number of Baseline Frames", fontsize=12)
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_title("AUC vs Calibration Duration", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.85, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_calibration_duration.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ablation_calibration_duration.png")

    # ── Figure 2: Per-Expression Comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    expressions = ["Algometer Pain", "Laser Pain", "Posed Pain"]
    short_names = ["Algometer", "Laser", "Posed"]

    for ax, expr, short in zip(axes, expressions, short_names):
        if expr not in expr_data:
            continue
        data = expr_data[expr]

        # Score distributions
        pain_scores = data["baud_pain_scores"]
        neutral_scores = data["baud_neutral_scores"]

        ax.hist(neutral_scores, bins=12, alpha=0.7, label="Neutral",
                color="#90CAF9", edgecolor="white")
        ax.hist(pain_scores, bins=12, alpha=0.7, label=f"{short} Pain",
                color="#EF5350", edgecolor="white")

        baud_auc = data["BAUD"]["auc"]
        gen_auc = data["Generic"]["auc"]
        ax.set_title(f"{short} Pain\nBAUD AUC={baud_auc:.3f} | Generic AUC={gen_auc:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("BAUD Pain Score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Expression Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_per_expression.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ablation_per_expression.png")

    # ── Figure 3: Per-Subject Benefit ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    benefits = [p["personalization_benefit"] for p in subj_data]
    base_vars = [p["baseline_variability"] for p in subj_data]
    subjects_sorted = [p["subject"] for p in subj_data]

    # Bar chart of personalization benefit
    colors = ["#4CAF50" if b > 0 else "#EF5350" for b in benefits]
    ax1.bar(range(len(benefits)), benefits, color=colors, alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Subjects (sorted by benefit)")
    ax1.set_ylabel("Personalization Benefit\n(BAUD separation − Generic separation)")
    ax1.set_title("Per-Subject Personalization Benefit", fontsize=13, fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Scatter: benefit vs baseline variability
    ax2.scatter(base_vars, benefits, c=benefits, cmap="RdYlGn", s=80,
                alpha=0.7, edgecolors="white")
    ax2.set_xlabel("Baseline Variability (facial expressiveness)", fontsize=11)
    ax2.set_ylabel("Personalization Benefit", fontsize=11)
    ax2.set_title("Benefit vs Expressiveness", fontsize=13, fontweight="bold")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_per_subject.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ablation_per_subject.png")

    # ── Figure 4: AU Ablation ──
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(au_data.keys())
    aucs = [au_data[n]["auc"] for n in names]
    f1s = [au_data[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, f1s, width, label="F1", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, aucs, width, label="AUC", color="#FF9800", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("AU Subset Ablation: Which AUs Matter?",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.3, 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i - width/2, f + 0.02, f"{f:.3f}", ha="center", fontsize=7)
        ax.text(i + width/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_au_subsets.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ablation_au_subsets.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  BAUD Ablation Experiments on PEMF")
    print("=" * 65)

    au_data = load_data()
    print(f"  Loaded {len(au_data)} subjects")

    # Run all experiments
    dur_data = exp1_calibration_duration(au_data)
    expr_data = exp2_per_expression(au_data)
    subj_data = exp3_per_subject_benefit(au_data)
    au_abl_data = exp4_au_ablation(au_data)

    # Generate all plots
    print(f"\n📈 Generating ablation plots...")
    plot_all_ablations(dur_data, expr_data, subj_data, au_abl_data, RESULTS_DIR)

    # Save comprehensive results
    results_path = os.path.join(RESULTS_DIR, "ablation_results.txt")
    with open(results_path, "w") as f:
        f.write("BAUD Ablation Experiments — PEMF Dataset\n")
        f.write("=" * 65 + "\n\n")

        f.write("Experiment 1: Calibration Duration\n")
        f.write("-" * 40 + "\n")
        durations, baud_aucs, baud_f1s, mah_aucs = dur_data
        for d, ba, bf, ma in zip(durations, baud_aucs, baud_f1s, mah_aucs):
            f.write(f"  {d:>3} frames: BAUD F1={bf:.4f}, AUC={ba:.4f} | "
                    f"Mahal AUC={ma:.4f}\n")

        f.write(f"\nExperiment 2: Per-Expression Analysis\n")
        f.write("-" * 40 + "\n")
        for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            if expr in expr_data:
                for method in ["BAUD", "Generic", "Mahalanobis"]:
                    m = expr_data[expr][method]
                    f.write(f"  {expr:<20} {method:<15} "
                            f"F1={m['f1']:.4f} AUC={m['auc']:.4f}\n")

        f.write(f"\nExperiment 4: AU Subset Ablation\n")
        f.write("-" * 40 + "\n")
        for name, m in au_abl_data.items():
            f.write(f"  {name:<30} F1={m['f1']:.4f} AUC={m['auc']:.4f}\n")

    print(f"  Saved: {results_path}")

    print(f"\n{'=' * 65}")
    print(f"  ✅ ALL ABLATION EXPERIMENTS COMPLETE")
    print(f"{'=' * 65}")
    print(f"\n  Results: {RESULTS_DIR}/")
    print(f"  ├── ablation_calibration_duration.png")
    print(f"  ├── ablation_per_expression.png")
    print(f"  ├── ablation_per_subject.png")
    print(f"  ├── ablation_au_subsets.png")
    print(f"  └── ablation_results.txt")
    print(f"\n  📤 Share all plots and the txt file!")


if __name__ == "__main__":
    main()
