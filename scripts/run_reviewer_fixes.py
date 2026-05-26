#!/usr/bin/env python3
"""
Reviewer Response Experiments
==============================
Fix 1 (W2): Supervised + Z-Score baseline on UNBC
Fix 2 (W6): 5-split variance on PEMF + per-subject UNBC box plot
Fix 3 (W4): Baseline contamination robustness test

Run on Colab:
    python scripts/run_reviewer_fixes.py
"""
import os, sys, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/results"
PEMF_CACHE = "/content/results/pemf_extracted_aus.npz"
UNBC_CACHE = "/content/results/unbc_extracted_aus.npz"
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
NUM_AUS = 41


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pemf():
    cached = np.load(PEMF_CACHE)
    au_data = {}
    for key in cached.files:
        subj, expr = key[:4], key[5:]
        if subj not in au_data:
            au_data[subj] = {}
        au_data[subj][expr] = cached[key]
    return au_data


def load_unbc():
    cached = np.load(UNBC_CACHE)
    au_data = {}
    for key in cached.files:
        parts = key.rsplit("_", 1)
        subj, level = parts[0], parts[1]
        if level == "pspi":
            continue
        if subj not in au_data:
            au_data[subj] = {}
        au_data[subj][level] = cached[key]
    return au_data


# ============================================================================
# SCORING METHODS
# ============================================================================

def baud_score_subject(neutral, test_aus):
    """BAUD with prior weights."""
    mean_b = np.mean(neutral, 0)
    std_b = np.maximum(np.std(neutral, 0), 1e-4)
    w = np.ones(test_aus.shape[1])
    for idx in PAIN_IDX:
        if idx < len(w): w[idx] = 3.0
    w /= w.sum()
    z = np.maximum((test_aus - mean_b) / std_b, 0)
    raw = np.array([np.dot(w, f) for f in z])
    return float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))


def generic_score(test_aus):
    return float(np.mean([np.mean(f[PAIN_IDX]) for f in test_aus]))


def mahalanobis_score(neutral, test_aus):
    fb = neutral[:, PAIN_IDX]
    mm = np.mean(fb, 0)
    mc = np.cov(fb, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
    mi = np.linalg.inv(mc)
    scores = []
    for f in test_aus[:, PAIN_IDX]:
        d = float(np.sqrt((f - mm) @ mi @ (f - mm)))
        scores.append(1.0 / (1.0 + np.exp(-d + 3.0)))
    return float(np.mean(scores))


def compute_metrics(scores, truths):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    preds = [1 if s > best_t else 0 for s in scores]
    try:
        auc = roc_auc_score(truths, scores)
    except:
        auc = 0.0
    return {"acc": accuracy_score(truths, preds), "f1": best_f1, "auc": auc}


# ============================================================================
# FIX 1: Supervised + Z-Score on UNBC (leave-one-subject-out)
# ============================================================================

def fix1_supervised_zscore(unbc_data):
    """
    The reviewer's key question: what if you give supervised models
    the SAME personalized z-score features that BAUD uses?
    This isolates whether BAUD's architecture matters or
    personalization alone is sufficient.
    """
    print("\n" + "=" * 70)
    print("  FIX 1 (W2): Supervised + Z-Score Baseline on UNBC")
    print("  'Give supervised models the same personalized features'")
    print("=" * 70)

    subjects = sorted([s for s in unbc_data if "0" in unbc_data[s]
                       and any(k != "0" for k in unbc_data[s])])

    methods = {
        "BAUD (Ours)": {"scores": [], "truths": [], "per_subj_auc": []},
        "Mahalanobis": {"scores": [], "truths": [], "per_subj_auc": []},
        "Generic (raw AUs)": {"scores": [], "truths": [], "per_subj_auc": []},
        "Sup. LR (raw AUs)": {"scores": [], "truths": [], "per_subj_auc": []},
        "Sup. MLP (raw AUs)": {"scores": [], "truths": [], "per_subj_auc": []},
        "Sup. LR + Z-Score": {"scores": [], "truths": [], "per_subj_auc": []},
        "Sup. MLP + Z-Score": {"scores": [], "truths": [], "per_subj_auc": []},
    }

    for hold_out in subjects:
        # This subject's data
        neutral_ho = unbc_data[hold_out]["0"]
        pain_keys = [k for k in unbc_data[hold_out] if k != "0"]
        if not pain_keys:
            continue
        pain_ho = np.concatenate([unbc_data[hold_out][k] for k in pain_keys])

        # Compute z-scores for held-out subject
        mean_ho = np.mean(neutral_ho, 0)
        std_ho = np.maximum(np.std(neutral_ho, 0), 1e-4)

        neutral_z = np.maximum((neutral_ho - mean_ho) / std_ho, 0)
        pain_z = np.maximum((pain_ho - mean_ho) / std_ho, 0)

        # Build training data from OTHER subjects
        train_raw_X, train_raw_y = [], []
        train_z_X, train_z_y = [], []

        for subj in subjects:
            if subj == hold_out:
                continue
            if "0" not in unbc_data[subj]:
                continue

            s_neutral = unbc_data[subj]["0"]
            s_pain_keys = [k for k in unbc_data[subj] if k != "0"]
            if not s_pain_keys:
                continue
            s_pain = np.concatenate([unbc_data[subj][k] for k in s_pain_keys])

            # Raw features
            for f in s_neutral:
                train_raw_X.append(f[PAIN_IDX])
                train_raw_y.append(0)
            for f in s_pain:
                train_raw_X.append(f[PAIN_IDX])
                train_raw_y.append(1)

            # Z-score normalized features (personalized per training subject)
            s_mean = np.mean(s_neutral, 0)
            s_std = np.maximum(np.std(s_neutral, 0), 1e-4)
            for f in s_neutral:
                z = np.maximum((f - s_mean) / s_std, 0)
                train_z_X.append(z[PAIN_IDX])
                train_z_y.append(0)
            for f in s_pain:
                z = np.maximum((f - s_mean) / s_std, 0)
                train_z_X.append(z[PAIN_IDX])
                train_z_y.append(1)

        train_raw_X = np.array(train_raw_X)
        train_raw_y = np.array(train_raw_y)
        train_z_X = np.array(train_z_X)
        train_z_y = np.array(train_z_y)

        # Test data for held-out subject
        test_raw_X = np.concatenate([neutral_ho[:, PAIN_IDX], pain_ho[:, PAIN_IDX]])
        test_raw_y = np.array([0]*len(neutral_ho) + [1]*len(pain_ho))
        test_z_X = np.concatenate([neutral_z[:, PAIN_IDX], pain_z[:, PAIN_IDX]])
        test_z_y = test_raw_y.copy()

        # ── Train supervised models ──
        # Raw AUs
        try:
            lr_raw = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            lr_raw.fit(train_raw_X, train_raw_y)
            lr_raw_probs = lr_raw.predict_proba(test_raw_X)[:, 1]
        except:
            lr_raw_probs = np.full(len(test_raw_y), 0.5)

        try:
            mlp_raw = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                                     random_state=42, early_stopping=True)
            mlp_raw.fit(train_raw_X, train_raw_y)
            mlp_raw_probs = mlp_raw.predict_proba(test_raw_X)[:, 1]
        except:
            mlp_raw_probs = np.full(len(test_raw_y), 0.5)

        # Z-Score normalized (PERSONALIZED features for supervised models!)
        try:
            lr_z = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            lr_z.fit(train_z_X, train_z_y)
            lr_z_probs = lr_z.predict_proba(test_z_X)[:, 1]
        except:
            lr_z_probs = np.full(len(test_z_y), 0.5)

        try:
            mlp_z = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                                   random_state=42, early_stopping=True)
            mlp_z.fit(train_z_X, train_z_y)
            mlp_z_probs = mlp_z.predict_proba(test_z_X)[:, 1]
        except:
            mlp_z_probs = np.full(len(test_z_y), 0.5)

        # ── Score all methods (subject-level) ──
        baud_s_n = baud_score_subject(neutral_ho, neutral_ho)
        baud_s_p = baud_score_subject(neutral_ho, pain_ho)
        gen_n = generic_score(neutral_ho)
        gen_p = generic_score(pain_ho)
        mah_n = mahalanobis_score(neutral_ho, neutral_ho)
        mah_p = mahalanobis_score(neutral_ho, pain_ho)

        # Subject-level scores for BAUD/Generic/Mahalanobis
        for name, n_score, p_score in [
            ("BAUD (Ours)", baud_s_n, baud_s_p),
            ("Mahalanobis", mah_n, mah_p),
            ("Generic (raw AUs)", gen_n, gen_p),
        ]:
            methods[name]["scores"].extend([n_score, p_score])
            methods[name]["truths"].extend([0, 1])
            try:
                auc = roc_auc_score([0, 1], [n_score, p_score])
            except:
                auc = 0.5
            methods[name]["per_subj_auc"].append(auc)

        # Frame-level scores for supervised models
        for name, probs, labels in [
            ("Sup. LR (raw AUs)", lr_raw_probs, test_raw_y),
            ("Sup. MLP (raw AUs)", mlp_raw_probs, test_raw_y),
            ("Sup. LR + Z-Score", lr_z_probs, test_z_y),
            ("Sup. MLP + Z-Score", mlp_z_probs, test_z_y),
        ]:
            mean_score_n = float(np.mean(probs[labels == 0]))
            mean_score_p = float(np.mean(probs[labels == 1]))
            methods[name]["scores"].extend([mean_score_n, mean_score_p])
            methods[name]["truths"].extend([0, 1])
            try:
                auc = roc_auc_score(labels, probs)
            except:
                auc = 0.5
            methods[name]["per_subj_auc"].append(auc)

    # ── Print Results ──
    print(f"\n{'='*75}")
    print(f"  UNBC-McMaster: Supervised + Z-Score Comparison (LOSO, {len(subjects)} subjects)")
    print(f"{'='*75}")
    print(f"  {'Method':<25} {'AUC':>8} {'F1':>8} {'Personalized?':>14} {'Labels':>8}")
    print(f"{'-'*75}")

    all_metrics = {}
    for name, data in methods.items():
        m = compute_metrics(data["scores"], data["truths"])
        m["per_subj_auc"] = data["per_subj_auc"]
        all_metrics[name] = m

        is_personal = "Yes" if "Z-Score" in name or name in ["BAUD (Ours)", "Mahalanobis"] else "No"
        has_labels = "Full" if "Sup." in name else ("Zero" if is_personal == "Yes" else "None")
        print(f"  {name:<25} {m['auc']:>8.4f} {m['f1']:>8.4f} {is_personal:>14} {has_labels:>8}")

    print(f"{'='*75}")

    # ── Key comparison ──
    baud_auc = all_metrics["BAUD (Ours)"]["auc"]
    sup_z_auc = all_metrics["Sup. LR + Z-Score"]["auc"]
    sup_raw_auc = all_metrics["Sup. LR (raw AUs)"]["auc"]

    print(f"\n  KEY FINDING:")
    print(f"  Supervised LR (raw AUs):    AUC = {sup_raw_auc:.4f}")
    print(f"  Supervised LR + Z-Score:    AUC = {sup_z_auc:.4f}  "
          f"({'↑' if sup_z_auc > sup_raw_auc else '↓'} "
          f"{abs(sup_z_auc - sup_raw_auc)*100:.1f}pp from personalization)")
    print(f"  BAUD (Ours):                AUC = {baud_auc:.4f}")

    if baud_auc > sup_z_auc:
        print(f"  → BAUD still outperforms supervised+z-score by "
              f"{(baud_auc-sup_z_auc)*100:.1f}pp")
        print(f"  → Both architecture AND personalization contribute")
    else:
        print(f"  → Supervised+z-score matches/exceeds BAUD")
        print(f"  → Personalization is the key contribution, not the architecture")

    return all_metrics


# ============================================================================
# FIX 1b: Per-Subject AUC Box Plot (UNBC)
# ============================================================================

def fix1b_per_subject_boxplot(all_metrics):
    """Box plot of per-subject AUC distributions on UNBC."""
    print("\n📈 Generating per-subject AUC box plot...")

    methods_to_plot = ["BAUD (Ours)", "Mahalanobis", "Sup. LR + Z-Score",
                       "Sup. LR (raw AUs)", "Generic (raw AUs)"]
    data = []
    labels = []
    for name in methods_to_plot:
        if name in all_metrics and all_metrics[name]["per_subj_auc"]:
            data.append(all_metrics[name]["per_subj_auc"])
            labels.append(name.replace(" (raw AUs)", "\n(raw)").replace(" + Z-Score", "\n+Z-Score"))

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#EF5350", "#9E9E9E"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-Subject AUC", fontsize=12)
    ax.set_title("UNBC-McMaster: Per-Subject AUC Distribution (LOSO)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.4, label="Random")

    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=20, color="black", zorder=5)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fix1_per_subject_boxplot.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fix1_per_subject_boxplot.png")


# ============================================================================
# FIX 2: 5-Split Variance on PEMF
# ============================================================================

def fix2_five_split_pemf(pemf_data):
    """Run 5 random splits on PEMF, report mean ± std."""
    print("\n" + "=" * 70)
    print("  FIX 2 (W6): 5-Split Variance on PEMF")
    print("=" * 70)

    subjects = sorted([s for s in pemf_data if "Neutral" in pemf_data[s]])
    seeds = [42, 123, 456, 789, 1024]

    method_names = ["BAUD (Ours)", "Mahalanobis", "One-Class SVM",
                    "Generic", "PSPI"]
    all_runs = {name: {"auc": [], "f1": [], "acc": []} for name in method_names}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(subjects))
        n_train = int(len(subjects) * 0.6)
        n_val = int(len(subjects) * 0.15)
        test_subjs = [subjects[i] for i in idx[n_train + n_val:]]

        # Run all methods on this split's test subjects
        methods = {name: {"scores": [], "truths": []} for name in method_names}

        for subj in test_subjs:
            neutral = pemf_data[subj].get("Neutral")
            if neutral is None:
                continue

            mean_b = np.mean(neutral, 0)
            std_b = np.maximum(np.std(neutral, 0), 1e-4)
            w = np.ones(neutral.shape[1])
            for i in PAIN_IDX:
                if i < len(w): w[i] = 3.0
            w /= w.sum()

            fb = neutral[:, PAIN_IDX]
            mm = np.mean(fb, 0)
            mc = np.cov(fb, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
            mi = np.linalg.inv(mc)

            try:
                ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
                ocsvm.fit(fb)
            except:
                ocsvm = None

            all_exprs = [("Neutral", 0)] + [(e, 1) for e in
                         ["Algometer Pain", "Laser Pain", "Posed Pain"]]

            for expr, label in all_exprs:
                aus = pemf_data[subj].get(expr)
                if aus is None:
                    continue

                # BAUD
                z = np.maximum((aus - mean_b) / std_b, 0)
                raw = np.array([np.dot(w, f) for f in z])
                baud_s = float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))

                # Generic
                gen_s = float(np.mean([np.mean(f[PAIN_IDX]) for f in aus]))

                # PSPI
                pspi_vals = [min((f[2]+max(f[4],f[5])+max(f[6],f[7])+(f[17] if len(f)>17 else 0))/2, 1) for f in aus]
                pspi_s = float(np.mean(pspi_vals))

                # Mahalanobis
                mah_vals = [float(1.0/(1.0+np.exp(-np.sqrt((f-mm)@mi@(f-mm))+3.0))) for f in aus[:, PAIN_IDX]]
                mah_s = float(np.mean(mah_vals))

                # OC-SVM
                if ocsvm:
                    svm_raw = ocsvm.decision_function(aus[:, PAIN_IDX])
                    svm_s = float(np.mean([1.0/(1.0+np.exp(r)) for r in svm_raw]))
                else:
                    svm_s = 0.5

                for name, score in [("BAUD (Ours)", baud_s), ("Generic", gen_s),
                                    ("PSPI", pspi_s), ("Mahalanobis", mah_s),
                                    ("One-Class SVM", svm_s)]:
                    methods[name]["scores"].append(score)
                    methods[name]["truths"].append(label)

        # Compute metrics for this split
        for name in method_names:
            m = compute_metrics(methods[name]["scores"], methods[name]["truths"])
            all_runs[name]["auc"].append(m["auc"])
            all_runs[name]["f1"].append(m["f1"])
            all_runs[name]["acc"].append(m["acc"])

    # ── Print Results ──
    print(f"\n{'='*75}")
    print(f"  PEMF: 5-Split Results (mean ± std)")
    print(f"{'='*75}")
    print(f"  {'Method':<20} {'Acc':>14} {'F1':>14} {'AUC':>14}")
    print(f"{'-'*75}")

    for name in method_names:
        acc_m, acc_s = np.mean(all_runs[name]["acc"]), np.std(all_runs[name]["acc"])
        f1_m, f1_s = np.mean(all_runs[name]["f1"]), np.std(all_runs[name]["f1"])
        auc_m, auc_s = np.mean(all_runs[name]["auc"]), np.std(all_runs[name]["auc"])
        print(f"  {name:<20} {acc_m:.3f}±{acc_s:.3f}  "
              f"{f1_m:.3f}±{f1_s:.3f}  {auc_m:.3f}±{auc_s:.3f}")

    print(f"{'='*75}")

    # ── Statistical significance (Wilcoxon) ──
    print(f"\n  Statistical Significance (Wilcoxon signed-rank test):")
    baud_aucs = all_runs["BAUD (Ours)"]["auc"]
    for baseline in ["Generic", "PSPI", "Mahalanobis"]:
        base_aucs = all_runs[baseline]["auc"]
        try:
            stat, p = wilcoxon(baud_aucs, base_aucs)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  BAUD vs {baseline:<15}: p={p:.4f} {sig}")
        except:
            print(f"  BAUD vs {baseline:<15}: insufficient samples for test")

    return all_runs


# ============================================================================
# FIX 3 (W4): Baseline Contamination Test
# ============================================================================

def fix3_baseline_contamination(pemf_data):
    """What happens if baseline contains some pain frames?"""
    print("\n" + "=" * 70)
    print("  FIX 3 (W4): Baseline Contamination Robustness")
    print("  'What if the baseline period contains mild pain?'")
    print("=" * 70)

    subjects = sorted([s for s in pemf_data if "Neutral" in pemf_data[s]])
    contamination_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

    results = {c: {"scores": [], "truths": []} for c in contamination_levels}
    rng = np.random.RandomState(42)

    for subj in subjects:
        neutral = pemf_data[subj].get("Neutral")
        if neutral is None:
            continue

        # Collect all pain frames for this subject
        pain_frames = []
        for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            p = pemf_data[subj].get(expr)
            if p is not None:
                pain_frames.append(p)
        if not pain_frames:
            continue
        all_pain = np.concatenate(pain_frames)

        for contamination in contamination_levels:
            # Create contaminated baseline
            n_contaminate = int(len(neutral) * contamination)
            if n_contaminate > 0 and len(all_pain) > 0:
                contam_idx = rng.choice(len(all_pain), min(n_contaminate, len(all_pain)), replace=False)
                contaminated_baseline = np.concatenate([
                    neutral,
                    all_pain[contam_idx]
                ])
            else:
                contaminated_baseline = neutral.copy()

            # Run BAUD with contaminated baseline
            # Score neutral (should be low)
            ns = baud_score_subject(contaminated_baseline, neutral)
            results[contamination]["scores"].append(ns)
            results[contamination]["truths"].append(0)

            # Score pain (should be high)
            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                p = pemf_data[subj].get(expr)
                if p is None:
                    continue
                ps = baud_score_subject(contaminated_baseline, p)
                results[contamination]["scores"].append(ps)
                results[contamination]["truths"].append(1)

    # Print results
    print(f"\n  {'Contamination':<15} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Δ AUC':>8}")
    print(f"  {'-'*50}")

    baseline_auc = None
    contam_aucs = []
    for c in contamination_levels:
        m = compute_metrics(results[c]["scores"], results[c]["truths"])
        if baseline_auc is None:
            baseline_auc = m["auc"]
        delta = m["auc"] - baseline_auc
        contam_aucs.append(m["auc"])
        print(f"  {c*100:>5.0f}%          {m['acc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {delta:>+8.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([c*100 for c in contamination_levels], contam_aucs,
            "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.set_xlabel("Baseline Contamination (%)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("BAUD Robustness to Baseline Contamination\n"
                 "(pain frames mixed into calibration period)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=baseline_auc, color="green", linestyle=":", alpha=0.5,
               label=f"Clean baseline AUC={baseline_auc:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fix3_contamination.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: fix3_contamination.png")

    return contamination_levels, contam_aucs


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  REVIEWER RESPONSE EXPERIMENTS")
    print("  Fix 1 (W2): Supervised + Z-Score")
    print("  Fix 2 (W6): 5-Split Variance")
    print("  Fix 3 (W4): Baseline Contamination")
    print("=" * 70)

    # Load data
    print("\n📂 Loading datasets...")
    pemf_data = load_pemf()
    print(f"  PEMF: {len(pemf_data)} subjects")

    unbc_data = None
    if os.path.exists(UNBC_CACHE):
        unbc_data = load_unbc()
        print(f"  UNBC: {len(unbc_data)} subjects")
    else:
        print("  UNBC: cache not found, skipping Fix 1")

    # ── Fix 1: Supervised + Z-Score ──
    if unbc_data:
        all_metrics = fix1_supervised_zscore(unbc_data)
        fix1b_per_subject_boxplot(all_metrics)

    # ── Fix 2: 5-Split Variance ──
    all_runs = fix2_five_split_pemf(pemf_data)

    # ── Fix 3: Baseline Contamination ──
    contamination_levels, contam_aucs = fix3_baseline_contamination(pemf_data)

    # ── Save all results ──
    results_file = f"{RESULTS_DIR}/reviewer_fixes_results.txt"
    with open(results_file, "w") as f:
        f.write("Reviewer Response Experiments\n")
        f.write("=" * 70 + "\n\n")
        f.write("Fix 1: See console output above\n")
        f.write("Fix 2: See console output above\n")
        f.write("Fix 3: Baseline Contamination\n")
        for c, a in zip(contamination_levels, contam_aucs):
            f.write(f"  {c*100:.0f}%: AUC={a:.4f}\n")
    print(f"\n  Saved: {results_file}")

    print(f"\n{'='*70}")
    print(f"  ✅ ALL REVIEWER FIXES COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Results: {RESULTS_DIR}/")
    print(f"  ├── fix1_per_subject_boxplot.png")
    print(f"  ├── fix3_contamination.png")
    print(f"  └── reviewer_fixes_results.txt")
    print(f"\n  📤 Share FULL console output + all plots!")


if __name__ == "__main__":
    main()
