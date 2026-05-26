#!/usr/bin/env python3
"""
Fix 4 (W3): Missing Recent Baselines

Implements simplified versions of:
1. DeepFaceLIFT (Rudovic et al., 2018) — expressiveness-aware personalization
2. ICU Relation Network (Chao et al., 2022) — baseline-vs-test comparison
3. Few-Shot Fine-Tuning — upper bound with K labeled pain examples per patient
4. Supervised + Z-Score (from Fix 1) — personalized supervised baseline
Plus all existing baselines for a complete comparison.

Run on Colab:
    python scripts/run_baselines_complete.py
"""
import os, sys, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/results"
PEMF_CACHE = "/content/results/pemf_extracted_aus.npz"
UNBC_CACHE = "/content/results/unbc_extracted_aus.npz"
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
SEED = 42


def load_dataset(cache_path, dataset_name):
    """Load cached AUs with correct key parsing."""
    cached = np.load(cache_path)
    data = {}

    if dataset_name == "pemf":
        for key in cached.files:
            subj, expr = key[:4], key[5:]
            if subj not in data:
                data[subj] = {}
            data[subj][expr] = cached[key]
    elif dataset_name == "unbc":
        for key in cached.files:
            if "_pspi" in key:
                continue
            subj = key.split("_")[0]
            label = key.split("_", 1)[1]
            if subj not in data:
                data[subj] = {}
            data[subj][label] = cached[key]

    return data


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
# BASELINE IMPLEMENTATIONS
# ============================================================================

class BAUDBaseline:
    """BAUD with prior weights (our method)."""
    name = "BAUD (Ours)"
    needs_labels = "Zero"
    personalized = True

    def score_subject(self, neutral, pain):
        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)
        w = np.ones(neutral.shape[1])
        for i in PAIN_IDX:
            if i < len(w): w[i] = 3.0
        w /= w.sum()
        z = np.maximum((pain - mean_b) / std_b, 0)
        raw = np.array([np.dot(w, f) for f in z])
        return float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))


class GenericBaseline:
    """Mean pain AU activation (no personalization)."""
    name = "Generic"
    needs_labels = "None"
    personalized = False

    def score_subject(self, neutral, test_aus):
        return float(np.mean([np.mean(f[PAIN_IDX]) for f in test_aus]))


class PSPIBaseline:
    """Clinical PSPI formula."""
    name = "PSPI"
    needs_labels = "None"
    personalized = False

    def score_subject(self, neutral, test_aus):
        vals = [min((f[2]+max(f[4],f[5])+max(f[6],f[7])+(f[17] if len(f)>17 else 0))/2, 1)
                for f in test_aus]
        return float(np.mean(vals))


class MahalanobisBaseline:
    """Mahalanobis distance from baseline centroid."""
    name = "Mahalanobis"
    needs_labels = "Zero"
    personalized = True

    def score_subject(self, neutral, test_aus):
        fb = neutral[:, PAIN_IDX]
        mm = np.mean(fb, 0)
        mc = np.cov(fb, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
        mi = np.linalg.inv(mc)
        scores = [1.0/(1.0+np.exp(-np.sqrt((f-mm)@mi@(f-mm))+3.0))
                  for f in test_aus[:, PAIN_IDX]]
        return float(np.mean(scores))


class OCSVMBaseline:
    """One-Class SVM on baseline features."""
    name = "One-Class SVM"
    needs_labels = "Zero"
    personalized = True

    def score_subject(self, neutral, test_aus):
        try:
            clf = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
            clf.fit(neutral[:, PAIN_IDX])
            raw = clf.decision_function(test_aus[:, PAIN_IDX])
            return float(np.mean([1.0/(1.0+np.exp(r)) for r in raw]))
        except:
            return 0.5


class DeepFaceLIFTBaseline:
    """
    Simplified DeepFaceLIFT (Rudovic et al., 2018).

    Core idea: compute per-subject expressiveness features
    (AU range, variability, asymmetry) from baseline and use them
    to modulate pain scoring. The original uses personalized
    multi-task learning; we implement the expressiveness-aware
    scoring component.

    Personalized: Yes (uses baseline expressiveness profile)
    Labels needed: Zero per patient (uses cross-subject training)
    """
    name = "DeepFaceLIFT*"
    needs_labels = "Zero"
    personalized = True

    def __init__(self):
        self.clf = None

    def compute_expressiveness(self, neutral_aus):
        """Compute expressiveness features from baseline."""
        # Per-AU expressiveness: range, std, mean
        au_mean = np.mean(neutral_aus, 0)
        au_std = np.std(neutral_aus, 0)
        au_range = np.ptp(neutral_aus, 0)  # max - min per AU

        # Aggregate expressiveness
        overall_expr = np.mean(au_std)
        pain_au_expr = np.mean(au_std[PAIN_IDX])

        return np.concatenate([
            au_mean[PAIN_IDX],      # 6 features: baseline AU means
            au_std[PAIN_IDX],       # 6 features: baseline AU variability
            au_range[PAIN_IDX],     # 6 features: baseline AU range
            [overall_expr],         # 1 feature: overall expressiveness
            [pain_au_expr],         # 1 feature: pain AU expressiveness
        ])  # Total: 20 features

    def train(self, all_subjects_data, neutral_key, pain_keys):
        """Train on all subjects with expressiveness features."""
        X, y = [], []
        for subj_data in all_subjects_data:
            neutral = subj_data[neutral_key]
            expr_feats = self.compute_expressiveness(neutral)

            # Neutral frames: low pain
            for f in neutral:
                features = np.concatenate([f[PAIN_IDX], expr_feats])
                X.append(features)
                y.append(0)

            # Pain frames: high pain
            for pk in pain_keys:
                if pk in subj_data:
                    for f in subj_data[pk]:
                        features = np.concatenate([f[PAIN_IDX], expr_feats])
                        X.append(features)
                        y.append(1)

        X, y = np.array(X), np.array(y)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=SEED)
        self.clf.fit(X, y)

    def score_subject(self, neutral, test_aus):
        if self.clf is None:
            return 0.5
        expr_feats = self.compute_expressiveness(neutral)
        X = np.array([np.concatenate([f[PAIN_IDX], expr_feats]) for f in test_aus])
        probs = self.clf.predict_proba(X)[:, 1]
        return float(np.mean(probs))


class RelationNetBaseline:
    """
    Simplified ICU Relation Network (Chao et al., 2022).

    Core idea: for each test frame, compute similarity to baseline
    frames. Pain frames should be dissimilar to baseline.
    The original uses a learned CNN relation module; we implement
    the comparison principle using learned distance in AU space.

    Personalized: Yes (compares to patient's own baseline)
    Labels needed: Full (cross-subject training)
    """
    name = "RelationNet*"
    needs_labels = "Full"
    personalized = True

    def __init__(self):
        self.clf = None

    def compute_relation_features(self, neutral, test_frame):
        """Compute relation features: test vs baseline comparison."""
        baseline_mean = np.mean(neutral[:, PAIN_IDX], 0)
        baseline_std = np.maximum(np.std(neutral[:, PAIN_IDX], 0), 1e-4)
        test_pain = test_frame[PAIN_IDX]

        # Difference features
        diff = test_pain - baseline_mean
        abs_diff = np.abs(diff)

        # Ratio features
        ratio = test_pain / (baseline_mean + 1e-4)

        # Distance
        z_score = diff / baseline_std
        euclidean = np.sqrt(np.sum(diff**2))
        cosine_sim = np.dot(test_pain, baseline_mean) / (
            np.linalg.norm(test_pain) * np.linalg.norm(baseline_mean) + 1e-8)

        return np.concatenate([
            diff,           # 6: signed difference
            abs_diff,       # 6: absolute difference
            z_score,        # 6: z-score difference
            [euclidean],    # 1: euclidean distance
            [cosine_sim],   # 1: cosine similarity
        ])  # Total: 20 features

    def train(self, all_subjects_data, neutral_key, pain_keys):
        """Train relation classifier across subjects."""
        X, y = [], []
        for subj_data in all_subjects_data:
            neutral = subj_data[neutral_key]

            for f in neutral:
                X.append(self.compute_relation_features(neutral, f))
                y.append(0)

            for pk in pain_keys:
                if pk in subj_data:
                    for f in subj_data[pk]:
                        X.append(self.compute_relation_features(neutral, f))
                        y.append(1)

        X, y = np.array(X), np.array(y)
        self.clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                  random_state=SEED, early_stopping=True)
        self.clf.fit(X, y)

    def score_subject(self, neutral, test_aus):
        if self.clf is None:
            return 0.5
        X = np.array([self.compute_relation_features(neutral, f) for f in test_aus])
        probs = self.clf.predict_proba(X)[:, 1]
        return float(np.mean(probs))


class FewShotBaseline:
    """
    Few-shot fine-tuning baseline.

    Uses K labeled pain examples per subject to fine-tune a classifier.
    This represents the upper bound of what's achievable with labels.

    Personalized: Yes
    Labels needed: K per patient
    """
    name = "Few-Shot (K=5)"
    needs_labels = "5/patient"
    personalized = True

    def __init__(self, k=5):
        self.k = k

    def score_subject(self, neutral, pain_aus):
        """Fine-tune on K pain examples + all neutral, score the rest."""
        rng = np.random.RandomState(SEED)

        if len(pain_aus) <= self.k:
            support_pain = pain_aus
            query_pain = pain_aus
        else:
            indices = rng.choice(len(pain_aus), self.k, replace=False)
            support_pain = pain_aus[indices]
            query_mask = np.ones(len(pain_aus), dtype=bool)
            query_mask[indices] = False
            query_pain = pain_aus[query_mask]

        # Train on neutral + K pain examples
        X_train = np.concatenate([neutral[:, PAIN_IDX], support_pain[:, PAIN_IDX]])
        y_train = np.array([0]*len(neutral) + [1]*len(support_pain))

        try:
            clf = LogisticRegression(max_iter=500, class_weight="balanced",
                                      random_state=SEED)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(query_pain[:, PAIN_IDX])[:, 1]
            return float(np.mean(probs))
        except:
            return 0.5


class SupervisedZScoreBaseline:
    """Supervised LR trained on z-score normalized features."""
    name = "Sup. LR+Z-Score"
    needs_labels = "Full"
    personalized = True

    def __init__(self):
        self.clf = None

    def train(self, all_subjects_data, neutral_key, pain_keys):
        X, y = [], []
        for subj_data in all_subjects_data:
            neutral = subj_data[neutral_key]
            mean_b = np.mean(neutral, 0)
            std_b = np.maximum(np.std(neutral, 0), 1e-4)

            for f in neutral:
                z = np.maximum((f - mean_b) / std_b, 0)
                X.append(z[PAIN_IDX]); y.append(0)
            for pk in pain_keys:
                if pk in subj_data:
                    for f in subj_data[pk]:
                        z = np.maximum((f - mean_b) / std_b, 0)
                        X.append(z[PAIN_IDX]); y.append(1)

        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=SEED)
        self.clf.fit(np.array(X), np.array(y))

    def score_subject(self, neutral, test_aus):
        if self.clf is None:
            return 0.5
        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)
        X = np.array([np.maximum((f - mean_b) / std_b, 0)[PAIN_IDX] for f in test_aus])
        probs = self.clf.predict_proba(X)[:, 1]
        return float(np.mean(probs))


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_on_dataset(data, dataset_name):
    """Run all baselines on a dataset."""
    if dataset_name == "pemf":
        neutral_key = "Neutral"
        pain_keys = ["Algometer Pain", "Laser Pain", "Posed Pain"]
    else:  # unbc
        neutral_key = "neutral"
        pain_keys = ["pain"]

    subjects = sorted([s for s in data
                       if neutral_key in data[s]
                       and any(pk in data[s] for pk in pain_keys)])

    print(f"\n  Evaluating on {len(subjects)} subjects...")

    # Initialize baselines
    simple_baselines = [
        BAUDBaseline(), GenericBaseline(), PSPIBaseline(),
        MahalanobisBaseline(), OCSVMBaseline(), FewShotBaseline(k=5),
    ]

    # Trainable baselines need LOSO
    trainable_baselines = [
        DeepFaceLIFTBaseline(),
        RelationNetBaseline(),
        SupervisedZScoreBaseline(),
    ]

    all_methods = {}

    # ── Simple baselines (no cross-subject training needed) ──
    for baseline in simple_baselines:
        scores, truths = [], []
        for subj in subjects:
            neutral = data[subj][neutral_key]

            # Score neutral
            s_n = baseline.score_subject(neutral, neutral)
            scores.append(s_n); truths.append(0)

            # Score pain
            for pk in pain_keys:
                if pk in data[subj]:
                    s_p = baseline.score_subject(neutral, data[subj][pk])
                    scores.append(s_p); truths.append(1)

        all_methods[baseline.name] = {
            **compute_metrics(scores, truths),
            "labels": baseline.needs_labels,
            "personal": baseline.personalized,
        }

    # ── Trainable baselines (LOSO) ──
    for baseline in trainable_baselines:
        scores, truths = [], []

        for hold_out in subjects:
            # Train on all other subjects
            train_data = [data[s] for s in subjects if s != hold_out]
            baseline.train(train_data, neutral_key, pain_keys)

            # Test on held-out subject
            neutral = data[hold_out][neutral_key]
            s_n = baseline.score_subject(neutral, neutral)
            scores.append(s_n); truths.append(0)

            for pk in pain_keys:
                if pk in data[hold_out]:
                    s_p = baseline.score_subject(neutral, data[hold_out][pk])
                    scores.append(s_p); truths.append(1)

        all_methods[baseline.name] = {
            **compute_metrics(scores, truths),
            "labels": baseline.needs_labels,
            "personal": baseline.personalized,
        }

    return all_methods


def print_results(all_methods, dataset_name):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"  COMPLETE BASELINE COMPARISON — {dataset_name}")
    print(f"{'='*80}")
    print(f"  {'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}  "
          f"{'Personal':>10} {'Labels':>12}")
    print(f"{'-'*80}")

    # Sort: personalized first, then by AUC
    sorted_methods = sorted(all_methods.items(),
                            key=lambda x: (-x[1]["personal"], -x[1]["auc"]))

    for name, m in sorted_methods:
        p = "Yes" if m["personal"] else "No"
        print(f"  {name:<25} {m['acc']:>8.4f} {m['f1']:>8.4f} {m['auc']:>8.4f}  "
              f"{p:>10} {m['labels']:>12}")

    print(f"{'='*80}")
    return sorted_methods


def plot_comparison(results_dict, save_path, title):
    """Bar chart comparing all methods."""
    # Sort by AUC
    sorted_items = sorted(results_dict.items(), key=lambda x: -x[1]["auc"])
    names = [n for n, _ in sorted_items]
    aucs = [m["auc"] for _, m in sorted_items]
    f1s = [m["f1"] for _, m in sorted_items]
    personalized = [m["personal"] for _, m in sorted_items]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.35

    colors_f1 = ["#1565C0" if p else "#90CAF9" for p in personalized]
    colors_auc = ["#E65100" if p else "#FFCC80" for p in personalized]

    ax.bar(x - width/2, f1s, width, color=colors_f1, alpha=0.85,
           edgecolor="white", label="F1 (dark=personalized)")
    ax.bar(x + width/2, aucs, width, color=colors_auc, alpha=0.85,
           edgecolor="white", label="AUC (dark=personalized)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0.3, 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i - width/2, f + 0.02, f"{f:.3f}", ha="center", fontsize=7)
        ax.text(i + width/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  FIX 4 (W3): Complete Baseline Comparison")
    print("  Including DeepFaceLIFT, Relation Network, Few-Shot")
    print("=" * 70)

    # ── PEMF ──
    if os.path.exists(PEMF_CACHE):
        print("\n📂 Loading PEMF...")
        pemf_data = load_dataset(PEMF_CACHE, "pemf")
        print(f"  {len(pemf_data)} subjects")

        pemf_results = evaluate_on_dataset(pemf_data, "pemf")
        print_results(pemf_results, "PEMF (68 subjects, lab pain)")
        plot_comparison(pemf_results,
                        f"{RESULTS_DIR}/fix4_pemf_all_baselines.png",
                        "PEMF: Complete Baseline Comparison (9 methods)")
    else:
        print("  PEMF cache not found, skipping")
        pemf_results = None

    # ── UNBC ──
    if os.path.exists(UNBC_CACHE):
        print("\n📂 Loading UNBC...")
        unbc_data = load_dataset(UNBC_CACHE, "unbc")
        print(f"  {len(unbc_data)} subjects")

        unbc_results = evaluate_on_dataset(unbc_data, "unbc")
        print_results(unbc_results, "UNBC-McMaster (24 subjects, clinical pain)")
        plot_comparison(unbc_results,
                        f"{RESULTS_DIR}/fix4_unbc_all_baselines.png",
                        "UNBC: Complete Baseline Comparison (9 methods)")
    else:
        print("  UNBC cache not found, skipping")
        unbc_results = None

    # ── Save combined results ──
    with open(f"{RESULTS_DIR}/fix4_all_baselines.txt", "w") as f:
        f.write("Complete Baseline Comparison\n")
        f.write("=" * 80 + "\n\n")

        if pemf_results:
            f.write("PEMF:\n")
            for name, m in sorted(pemf_results.items(), key=lambda x: -x[1]["auc"]):
                p = "Yes" if m["personal"] else "No"
                f.write(f"  {name:<25} AUC={m['auc']:.4f} F1={m['f1']:.4f} "
                        f"Personal={p} Labels={m['labels']}\n")

        if unbc_results:
            f.write("\nUNBC:\n")
            for name, m in sorted(unbc_results.items(), key=lambda x: -x[1]["auc"]):
                p = "Yes" if m["personal"] else "No"
                f.write(f"  {name:<25} AUC={m['auc']:.4f} F1={m['f1']:.4f} "
                        f"Personal={p} Labels={m['labels']}\n")

    print(f"\n  Saved: fix4_all_baselines.txt")

    print(f"\n{'='*70}")
    print(f"  ✅ COMPLETE BASELINE COMPARISON DONE")
    print(f"{'='*70}")
    print(f"  📤 Share console output + both plots!")


if __name__ == "__main__":
    main()
