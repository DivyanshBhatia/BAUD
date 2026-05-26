#!/usr/bin/env python3
"""
CRITICAL EXPERIMENT: Baseline Estimation Heuristics for Protocol C

The problem: Using first-K chronological frames as baseline gives BAUD 0.680 AUC,
below Generic's 0.697. The frames may include pain, settling-in artifacts, or
unrepresentative expressions.

Solution: Label-blind heuristics to SELECT better baseline frames from the first K.
All heuristics are deployment-realistic (no PSPI label access).

Heuristics:
1. Lowest-activity window: Find the W-frame window with lowest mean AU activation
2. Lowest-variance frames: Select frames with lowest total AU variance
3. Robust quantile: Use 25th percentile of each AU as baseline center
4. Cluster-and-select: K-means on K frames, pick lowest-activity cluster
5. Rolling median baseline: Use rolling median over first K frames
6. Outlier rejection: Remove frames with AU values > 2 std from mean

Run on Colab after run_protocol_c.py has finished:
    python scripts/run_baseline_fix.py
"""
import os, sys, time, glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES_DIR = "/content/Images/Images"
LABELS_DIR = "/content/Frame_Labels/Frame_Labels/PSPI"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
BATCH_SIZE = 64
SEED = 42

au_transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    sys.path.insert(0, OPENGRAPHAU_DIR)
    from model.MEFL import MEFARG
    model = MEFARG(num_main_classes=27, num_sub_classes=14, backbone="resnet50")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    sd = OrderedDict((k.replace("module.", ""), v)
                     for k, v in ckpt.get("state_dict", ckpt).items())
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    return model


def extract_aus(model, paths):
    all_aus = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = []
        for p in paths[i:i+BATCH_SIZE]:
            try: batch.append(au_transform(Image.open(p).convert("RGB")))
            except: batch.append(torch.zeros(3, 224, 224))
        t = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            out = model(t)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            all_aus.append(torch.sigmoid(pred).cpu().numpy())
    return np.concatenate(all_aus)


def get_chronological_frames():
    subjects = {}
    for subj_folder in sorted(os.listdir(IMAGES_DIR)):
        subj_id = subj_folder.split("-")[0]
        subj_img = os.path.join(IMAGES_DIR, subj_folder)
        subj_lbl = os.path.join(LABELS_DIR, subj_folder)
        if not os.path.isdir(subj_img) or not os.path.isdir(subj_lbl):
            continue
        frames = []
        for seq in sorted(os.listdir(subj_img)):
            seq_img = os.path.join(subj_img, seq)
            seq_lbl = os.path.join(subj_lbl, seq)
            if not os.path.isdir(seq_img) or not os.path.isdir(seq_lbl):
                continue
            for img_file in sorted(os.listdir(seq_img)):
                if not img_file.endswith(".png"): continue
                lbl_file = os.path.join(seq_lbl, img_file.replace(".png","")+"_facs.txt")
                if os.path.exists(lbl_file):
                    try:
                        with open(lbl_file) as f: pspi = float(f.read().strip())
                        frames.append({"path": os.path.join(seq_img, img_file), "pspi": pspi})
                    except: pass
        if frames: subjects[subj_id] = frames
    return subjects


# ============================================================================
# BASELINE ESTIMATION HEURISTICS (all label-blind)
# ============================================================================

def heuristic_naive(aus_first_k):
    """Naive: use all first-K frames as-is (current Protocol C)."""
    return np.mean(aus_first_k, 0), np.maximum(np.std(aus_first_k, 0), 1e-4)

def heuristic_lowest_activity_window(aus_first_k, window=10):
    """Find the W-frame window with lowest mean AU activation."""
    if len(aus_first_k) <= window:
        return heuristic_naive(aus_first_k)
    best_score, best_idx = np.inf, 0
    for i in range(len(aus_first_k) - window + 1):
        w = aus_first_k[i:i+window]
        score = np.mean(w[:, PAIN_IDX])  # Mean pain-AU activity
        if score < best_score:
            best_score, best_idx = score, i
    selected = aus_first_k[best_idx:best_idx+window]
    return np.mean(selected, 0), np.maximum(np.std(selected, 0), 1e-4)

def heuristic_lowest_variance(aus_first_k, keep_frac=0.5):
    """Select the fraction of frames with lowest total AU variance."""
    n_keep = max(3, int(len(aus_first_k) * keep_frac))
    # Per-frame deviation from mean
    mean_all = np.mean(aus_first_k, 0)
    deviations = np.array([np.sum((f - mean_all)**2) for f in aus_first_k])
    idx = np.argsort(deviations)[:n_keep]
    selected = aus_first_k[idx]
    return np.mean(selected, 0), np.maximum(np.std(selected, 0), 1e-4)

def heuristic_robust_quantile(aus_first_k):
    """Use 25th percentile as baseline center, MAD as spread."""
    center = np.percentile(aus_first_k, 25, axis=0)
    mad = np.maximum(np.median(np.abs(aus_first_k - center), axis=0), 1e-4)
    return center, 1.4826 * mad

def heuristic_cluster_select(aus_first_k, n_clusters=2):
    """K-means into clusters, use lowest-activity cluster as baseline."""
    if len(aus_first_k) < n_clusters * 3:
        return heuristic_naive(aus_first_k)
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=5)
    labels = km.fit_predict(aus_first_k[:, PAIN_IDX])
    # Pick cluster with lowest mean pain-AU activity
    best_cluster = min(range(n_clusters),
                       key=lambda c: np.mean(aus_first_k[labels==c][:, PAIN_IDX]))
    selected = aus_first_k[labels == best_cluster]
    return np.mean(selected, 0), np.maximum(np.std(selected, 0), 1e-4)

def heuristic_outlier_rejection(aus_first_k, threshold=1.5):
    """Remove frames with AU values > threshold std from mean."""
    mean_all = np.mean(aus_first_k, 0)
    std_all = np.maximum(np.std(aus_first_k, 0), 1e-4)
    # Per-frame max z-score
    max_z = np.array([np.max(np.abs((f - mean_all) / std_all)) for f in aus_first_k])
    keep = max_z < threshold
    if np.sum(keep) < 3:
        keep = np.argsort(max_z)[:max(3, len(aus_first_k)//2)]
        selected = aus_first_k[keep]
    else:
        selected = aus_first_k[keep]
    return np.mean(selected, 0), np.maximum(np.std(selected, 0), 1e-4)


# ============================================================================
# BAUD SCORING
# ============================================================================

def score_baud(mean_b, std_b, test_aus):
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()
    scores = []
    for f in test_aus:
        z = np.maximum((f - mean_b) / std_b, 0)
        raw = np.dot(w, z)
        scores.append(1.0 / (1.0 + np.exp(-raw + 2.0)))
    return np.array(scores)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 75)
    print("  BASELINE ESTIMATION HEURISTICS FOR PROTOCOL C")
    print("  Can we push 0.680 → 0.725+ with label-blind frame selection?")
    print("=" * 75)

    model = load_model()
    print("  ✅ Model loaded")

    chrono = get_chronological_frames()
    print(f"  {len(chrono)} subjects")

    K = 50  # Use first 50 frames for baseline estimation
    heuristics = {
        "Naive (all K)": heuristic_naive,
        "Low-activity window": heuristic_lowest_activity_window,
        "Low-variance select": heuristic_lowest_variance,
        "Robust quantile": heuristic_robust_quantile,
        "Cluster & select": heuristic_cluster_select,
        "Outlier rejection": heuristic_outlier_rejection,
    }

    results = {name: [] for name in heuristics}
    results["Generic"] = []
    results["Full neutral (ref)"] = []

    t0 = time.time()
    for idx, (subj_id, frames) in enumerate(sorted(chrono.items())):
        aus = extract_aus(model, [f["path"] for f in frames])
        pspis = np.array([f["pspi"] for f in frames])
        labels = (pspis > 0).astype(int)

        if sum(labels) < 5 or sum(1-labels) < 5:
            continue

        first_k = aus[:K]
        test_aus = aus[K:]
        test_labels = labels[K:]
        if len(set(test_labels)) < 2:
            continue

        # Full neutral baseline (reference)
        neutral_aus = aus[labels == 0]
        mean_ref, std_ref = np.mean(neutral_aus, 0), np.maximum(np.std(neutral_aus, 0), 1e-4)
        scores_ref = score_baud(mean_ref, std_ref, test_aus)
        try: results["Full neutral (ref)"].append(roc_auc_score(test_labels, scores_ref))
        except: results["Full neutral (ref)"].append(0.5)

        # Generic
        gen_scores = np.array([np.mean(f[PAIN_IDX]) for f in test_aus])
        try: results["Generic"].append(roc_auc_score(test_labels, gen_scores))
        except: results["Generic"].append(0.5)

        # Each heuristic
        for name, heuristic_fn in heuristics.items():
            mean_b, std_b = heuristic_fn(first_k)
            scores = score_baud(mean_b, std_b, test_aus)
            try: results[name].append(roc_auc_score(test_labels, scores))
            except: results[name].append(0.5)

        if (idx+1) % 5 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    # ── Results ──
    print(f"\n{'='*75}")
    print(f"  BASELINE ESTIMATION HEURISTIC RESULTS (K={K}, Protocol C)")
    print(f"{'='*75}")
    print(f"  {'Heuristic':<25} {'Mean AUC':>10} {'Std':>8} {'vs Naive':>10} {'vs Generic':>12}")
    print(f"  {'-'*70}")

    naive_mean = np.mean(results["Naive (all K)"])
    generic_mean = np.mean(results["Generic"])

    sorted_results = sorted(results.items(), key=lambda x: -np.mean(x[1]))
    for name, aucs in sorted_results:
        m = np.mean(aucs)
        s = np.std(aucs)
        vs_naive = m - naive_mean
        vs_generic = m - generic_mean
        marker = " ✓" if m > generic_mean else ""
        print(f"  {name:<25} {m:>10.4f} {s:>8.4f} {vs_naive:>+10.4f} {vs_generic:>+12.4f}{marker}")

    print(f"{'='*75}")

    # Significance tests
    print(f"\n  Significance tests (Wilcoxon):")
    naive_aucs = results["Naive (all K)"]
    generic_aucs = results["Generic"]
    for name, aucs in sorted_results:
        if name in ["Generic", "Full neutral (ref)", "Naive (all K)"]:
            continue
        n = min(len(aucs), len(generic_aucs))
        if n >= 5:
            try:
                _, p = wilcoxon(aucs[:n], generic_aucs[:n])
                sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
                print(f"    {name:<25} vs Generic: p={p:.4f} {sig}")
            except:
                pass

    # Per-subject analysis: which subjects improve most?
    print(f"\n  Per-subject: Naive vs Best Heuristic")
    best_heuristic = max(heuristics.keys(), key=lambda n: np.mean(results[n]))
    best_aucs = results[best_heuristic]
    print(f"  Best heuristic: {best_heuristic}")
    improved = sum(1 for b, n in zip(best_aucs, naive_aucs) if b > n)
    degraded = sum(1 for b, n in zip(best_aucs, naive_aucs) if b < n)
    print(f"  Improved: {improved}/{len(naive_aucs)} subjects")
    print(f"  Degraded: {degraded}/{len(naive_aucs)} subjects")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = [name for name, _ in sorted_results if name != "Full neutral (ref)"]
    means = [np.mean(results[n]) for n in methods]
    colors = ["#4CAF50" if np.mean(results[n]) > generic_mean else "#EF5350"
              for n in methods]
    bars = ax.bar(range(len(methods)), means, color=colors, alpha=0.8,
                  edgecolor="white")
    ax.axhline(y=generic_mean, color="black", linestyle="--", alpha=0.5,
               label=f"Generic={generic_mean:.3f}")
    ax.axhline(y=np.mean(results["Full neutral (ref)"]), color="green",
               linestyle=":", alpha=0.5,
               label=f"Full neutral={np.mean(results['Full neutral (ref)']):.3f}")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Per-Subject AUC", fontsize=12)
    ax.set_title(f"Protocol C Baseline Heuristics (K={K})\n"
                 f"Green = beats Generic, Red = below Generic",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.5, 0.85)
    for i, m in enumerate(means):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/baseline_heuristics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: baseline_heuristics.png")

    print(f"\n{'='*75}")
    print(f"  ✅ BASELINE HEURISTIC EXPERIMENT COMPLETE")
    print(f"{'='*75}")
    print(f"  📤 Share console output + plot!")


if __name__ == "__main__":
    main()
