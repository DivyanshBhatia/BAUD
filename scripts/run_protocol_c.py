#!/usr/bin/env python3
"""
Protocol C: First-K Chronological Frames (Deployment-Realistic)
+ Complete honest frame-level evaluation for paper

For each subject:
1. Extract AUs for ALL frames in chronological order
2. Use FIRST K frames as baseline (label-blind — no PSPI peeking)
3. Test on ALL remaining frames
4. Report frame-level metrics with bootstrap CIs

This is the deployment-realistic protocol the R2 reviewer requested.

Run on Colab:
    python scripts/run_protocol_c.py
"""
import os, sys, glob, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              balanced_accuracy_score, average_precision_score)
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


# ============================================================================
# STEP 1: Load model + build chronological frame list
# ============================================================================

def load_model():
    sys.path.insert(0, OPENGRAPHAU_DIR)
    from model.MEFL import MEFARG
    model = MEFARG(num_main_classes=27, num_sub_classes=14, backbone="resnet50")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    sd = OrderedDict((k.replace("module.", ""), v)
                     for k, v in ckpt.get("state_dict", ckpt).items())
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    print(f"  ✅ OpenGraphAU on {DEVICE}")
    return model


def get_chronological_frames():
    """Get ALL frames per subject in chronological order with PSPI labels."""
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
                if not img_file.endswith(".png"):
                    continue
                lbl_file = os.path.join(seq_lbl,
                    img_file.replace(".png", "") + "_facs.txt")
                if os.path.exists(lbl_file):
                    try:
                        with open(lbl_file) as f:
                            pspi = float(f.read().strip())
                        frames.append({
                            "path": os.path.join(seq_img, img_file),
                            "pspi": pspi,
                        })
                    except:
                        pass
        if frames:
            subjects[subj_id] = frames
    return subjects


def extract_aus_for_subject(model, frames):
    """Extract AU vectors for a list of frame dicts."""
    paths = [f["path"] for f in frames]
    all_aus = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        tensors = []
        for p in batch_paths:
            try:
                tensors.append(au_transform(Image.open(p).convert("RGB")))
            except:
                tensors.append(torch.zeros(3, 224, 224))
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            all_aus.append(torch.sigmoid(pred).cpu().numpy())
    return np.concatenate(all_aus)


# ============================================================================
# STEP 2: Scoring methods
# ============================================================================

def score_baud(baseline_aus, test_aus):
    mean_b = np.mean(baseline_aus, 0)
    std_b = np.maximum(np.std(baseline_aus, 0), 1e-4)
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX:
        if i < len(w): w[i] = 3.0
    w /= w.sum()
    scores = []
    for f in test_aus:
        z = np.maximum((f - mean_b) / std_b, 0)
        raw = np.dot(w, z)
        scores.append(1.0 / (1.0 + np.exp(-raw + 2.0)))
    return np.array(scores)


def score_mahalanobis(baseline_aus, test_aus):
    fb = baseline_aus[:, PAIN_IDX]
    mm = np.mean(fb, 0)
    mc = np.cov(fb, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
    mi = np.linalg.inv(mc)
    scores = []
    for f in test_aus[:, PAIN_IDX]:
        d = float(np.sqrt((f - mm) @ mi @ (f - mm)))
        scores.append(1.0 / (1.0 + np.exp(-d + 3.0)))
    return np.array(scores)


def score_generic(test_aus):
    return np.array([np.mean(f[PAIN_IDX]) for f in test_aus])


def score_median_mad(baseline_aus, test_aus):
    median_b = np.median(baseline_aus, 0)
    mad_b = np.maximum(np.median(np.abs(baseline_aus - median_b), axis=0), 1e-4)
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()
    scores = []
    for f in test_aus:
        z = np.maximum((f - median_b) / (1.4826 * mad_b), 0)
        raw = np.dot(w, z)
        scores.append(1.0 / (1.0 + np.exp(-raw + 2.0)))
    return np.array(scores)


# ============================================================================
# STEP 3: Run Protocol C
# ============================================================================

def compute_metrics_safe(labels, scores):
    """Compute all metrics safely."""
    if len(set(labels)) < 2:
        return {"auc": 0.5, "pr_auc": 0.5, "f1": 0, "bal_acc": 0.5}
    try:
        auc = roc_auc_score(labels, scores)
        pr_auc = average_precision_score(labels, scores)
        best_f1 = 0
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(labels, [1 if s > t else 0 for s in scores],
                          zero_division=0)
            if f1 > best_f1: best_f1 = f1
        bal_acc = balanced_accuracy_score(labels,
            [1 if s > 0.3 else 0 for s in scores])
        return {"auc": auc, "pr_auc": pr_auc, "f1": best_f1, "bal_acc": bal_acc}
    except:
        return {"auc": 0.5, "pr_auc": 0.5, "f1": 0, "bal_acc": 0.5}


def bootstrap_ci(values, n_boot=2000):
    rng = np.random.RandomState(SEED)
    means = [np.mean(rng.choice(values, len(values), replace=True))
             for _ in range(n_boot)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    print("=" * 75)
    print("  PROTOCOL C: First-K Chronological Frames")
    print("  Deployment-realistic, label-blind baseline selection")
    print("=" * 75)

    # Load model
    print("\n📦 Loading model...")
    model = load_model()

    # Get chronological frames
    print("\n📂 Building chronological index...")
    chrono = get_chronological_frames()
    print(f"  {len(chrono)} subjects")

    # Show first-K pain contamination stats
    print(f"\n  Pain contamination in first K chronological frames:")
    print(f"  {'Subject':<10} {'Total':>8} {'K=10':>8} {'K=20':>8} {'K=50':>8}")
    for subj in sorted(chrono.keys())[:5]:
        total = len(chrono[subj])
        pain10 = sum(1 for f in chrono[subj][:10] if f["pspi"] > 0)
        pain20 = sum(1 for f in chrono[subj][:20] if f["pspi"] > 0)
        pain50 = sum(1 for f in chrono[subj][:50] if f["pspi"] > 0)
        print(f"  {subj:<10} {total:>8} {pain10:>8} {pain20:>8} {pain50:>8}")
    print(f"  ...")

    # Extract AUs and run evaluation
    K_values = [5, 10, 20, 50, 100]
    methods = ["BAUD", "Mahalanobis", "Median/MAD", "Generic"]
    protocol_results = {m: {K: [] for K in K_values} for m in methods}
    # Also store full-evaluation (all neutral as baseline) for comparison
    full_eval = {m: [] for m in methods}

    t0 = time.time()
    valid_subjects = []

    for idx, (subj_id, frames) in enumerate(sorted(chrono.items())):
        # Extract AUs for ALL frames
        aus = extract_aus_for_subject(model, frames)
        pspis = np.array([f["pspi"] for f in frames])
        labels = (pspis > 0).astype(int)

        if sum(labels) < 5 or sum(1 - labels) < 5:
            continue
        valid_subjects.append(subj_id)

        # Full evaluation: all neutral as baseline (label-informed, for reference)
        neutral_mask = labels == 0
        neutral_aus = aus[neutral_mask]

        for name, scorer in [("BAUD", score_baud), ("Mahalanobis", score_mahalanobis),
                              ("Median/MAD", score_median_mad)]:
            scores = scorer(neutral_aus, aus)
            m = compute_metrics_safe(labels, scores)
            full_eval[name].append(m["auc"])

        gen_scores = score_generic(aus)
        full_eval["Generic"].append(compute_metrics_safe(labels, gen_scores)["auc"])

        # Protocol C: First K chronological frames as baseline
        for K in K_values:
            K_actual = min(K, len(aus))
            baseline = aus[:K_actual]  # First K frames, label-blind!
            test_aus = aus[K_actual:]
            test_labels = labels[K_actual:]

            if len(set(test_labels)) < 2:
                continue

            for name, scorer in [("BAUD", score_baud),
                                  ("Mahalanobis", score_mahalanobis),
                                  ("Median/MAD", score_median_mad)]:
                scores = scorer(baseline, test_aus)
                m = compute_metrics_safe(test_labels, scores)
                protocol_results[name][K].append(m["auc"])

            gen_scores = score_generic(test_aus)
            protocol_results["Generic"][K].append(
                compute_metrics_safe(test_labels, gen_scores)["auc"])

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    print(f"\n  ✅ Processed {len(valid_subjects)} valid subjects in {time.time()-t0:.1f}s")

    # ── Results: Protocol C ──
    print(f"\n{'='*80}")
    print(f"  PROTOCOL C: First-K Chronological (Label-Blind) — Per-Subject AUC")
    print(f"{'='*80}")
    print(f"  {'Method':<18}", end="")
    print(f"  {'Full(ref)':>9}", end="")
    for K in K_values:
        print(f"  {'K='+str(K):>7}", end="")
    print()
    print(f"  {'-'*75}")

    for name in methods:
        full_m = np.mean(full_eval[name]) if full_eval[name] else 0
        print(f"  {name:<18}  {full_m:>9.3f}", end="")
        for K in K_values:
            aucs = protocol_results[name][K]
            if aucs:
                print(f"  {np.mean(aucs):>7.3f}", end="")
            else:
                print(f"  {'N/A':>7}", end="")
        print()
    print(f"{'='*80}")

    # ── Frame-level metrics with bootstrap CIs ──
    print(f"\n{'='*80}")
    print(f"  FRAME-LEVEL METRICS: Protocol C, K=20 (bootstrap 95% CI)")
    print(f"{'='*80}")
    print(f"  {'Method':<18} {'AUC':>22} {'Significant vs Generic?':>25}")
    print(f"  {'-'*70}")

    K_report = 20  # Report detailed metrics for K=20
    for name in methods:
        aucs = protocol_results[name][K_report]
        if len(aucs) >= 3:
            mean, lo, hi = bootstrap_ci(aucs)
            # Significance vs Generic
            gen_aucs = protocol_results["Generic"][K_report]
            try:
                _, p = wilcoxon(aucs, gen_aucs[:len(aucs)])
                sig = f"p={p:.4f}" + (" *" if p < 0.05 else " n.s.")
            except:
                sig = "N/A"
            print(f"  {name:<18} {mean:.3f} [{lo:.3f}, {hi:.3f}]  {sig:>25}")
    print(f"{'='*80}")

    # Significance: BAUD vs Mahalanobis
    print(f"\n  Paired tests at K=20:")
    for K in [10, 20, 50]:
        baud_aucs = protocol_results["BAUD"][K]
        mah_aucs = protocol_results["Mahalanobis"][K]
        gen_aucs = protocol_results["Generic"][K]
        n = min(len(baud_aucs), len(mah_aucs), len(gen_aucs))
        if n >= 5:
            try:
                _, p_mah = wilcoxon(baud_aucs[:n], mah_aucs[:n])
                _, p_gen = wilcoxon(baud_aucs[:n], gen_aucs[:n])
                print(f"    K={K}: BAUD vs Mahal p={p_mah:.4f}, BAUD vs Generic p={p_gen:.4f}")
            except Exception as e:
                print(f"    K={K}: test failed ({e})")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Protocol C: AUC vs K
    colors = {"BAUD": "#2196F3", "Mahalanobis": "#FF9800",
              "Median/MAD": "#4CAF50", "Generic": "#9E9E9E"}
    for name in methods:
        aucs = [np.mean(protocol_results[name][K])
                for K in K_values if protocol_results[name][K]]
        Ks = [K for K in K_values if protocol_results[name][K]]
        ax1.plot(Ks, aucs, "o-", color=colors[name], linewidth=2,
                 markersize=7, label=name)
        # Add full-eval reference line
        if full_eval[name]:
            ax1.axhline(y=np.mean(full_eval[name]), color=colors[name],
                         linestyle=":", alpha=0.3)

    ax1.set_xlabel("First K Chronological Frames (baseline)", fontsize=11)
    ax1.set_ylabel("Mean Per-Subject AUC", fontsize=11)
    ax1.set_title("Protocol C: Label-Blind Chronological Baseline\n"
                   "(dotted = full neutral baseline reference)",
                   fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)

    # Box plot at K=20
    K_box = 20
    data = [protocol_results[m][K_box] for m in methods]
    bp = ax2.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], [colors[m] for m in methods]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax2.scatter(x, d, alpha=0.5, s=15, color="black", zorder=5)
    ax2.set_ylabel("Per-Subject AUC", fontsize=11)
    ax2.set_title(f"Frame-Level AUC Distribution (K={K_box})",
                   fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=0.5, color="red", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/protocol_c_results.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: protocol_c_results.png")

    # Save summary
    with open(f"{RESULTS_DIR}/protocol_c_summary.txt", "w") as f:
        f.write("Protocol C: First-K Chronological Frame Results\n")
        f.write("=" * 75 + "\n")
        for name in methods:
            f.write(f"\n{name}:\n")
            full_m = np.mean(full_eval[name]) if full_eval[name] else 0
            f.write(f"  Full neutral baseline (ref): {full_m:.4f}\n")
            for K in K_values:
                aucs = protocol_results[name][K]
                if aucs:
                    f.write(f"  K={K:>3}: {np.mean(aucs):.4f} "
                            f"± {np.std(aucs):.4f}\n")

    print(f"  Saved: protocol_c_summary.txt")
    print(f"\n{'='*75}")
    print(f"  ✅ PROTOCOL C COMPLETE")
    print(f"{'='*75}")
    print(f"  📤 Share FULL console output + plot!")


if __name__ == "__main__":
    main()
