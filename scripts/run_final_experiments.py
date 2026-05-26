#!/usr/bin/env python3
"""
Two highest-impact experiments for AAAI submission:
1. Online Baseline Adaptation — self-improving baseline during monitoring
2. Per-subject P1→P2 scatter — visual proof of bottleneck

Run on Colab:
    python scripts/run_final_experiments.py
"""
import os, sys, time, numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
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
BATCH_SIZE = 64; K = 50; W = 10; SEED = 42

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
        if not os.path.isdir(subj_img) or not os.path.isdir(subj_lbl): continue
        frames = []
        for seq in sorted(os.listdir(subj_img)):
            seq_img = os.path.join(subj_img, seq)
            seq_lbl = os.path.join(subj_lbl, seq)
            if not os.path.isdir(seq_img) or not os.path.isdir(seq_lbl): continue
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

def get_lowact_window(first_k):
    if len(first_k) <= W: return first_k
    best_s, best_i = np.inf, 0
    for i in range(len(first_k) - W + 1):
        s = np.mean(first_k[i:i+W][:, PAIN_IDX])
        if s < best_s: best_s, best_i = s, i
    return first_k[best_i:best_i+W]

def score_baud_frame(mean_b, std_b, frame):
    w = np.ones(len(frame))
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()
    z = np.maximum((frame - mean_b) / std_b, 0)
    raw = np.dot(w, z)
    return 1.0 / (1.0 + np.exp(-raw + 2.0))


# ============================================================================
# EXPERIMENT 1: Online Baseline Adaptation
# ============================================================================

def run_online_adaptation(model, chrono):
    """
    After initial K-frame calibration, update baseline using frames
    the model itself considers non-pain (score < threshold).
    
    This is self-supervised: no labels needed, just the model's own
    confidence that a frame is neutral.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Online Baseline Adaptation")
    print("  Self-improving baseline during monitoring")
    print("=" * 70)

    thresholds = [0.2, 0.3, 0.4, 0.5]  # Update baseline if score < threshold
    alphas = [0.01, 0.05, 0.1]  # EMA decay rates

    results = {}
    # Reference methods
    results["P1 (clean baseline)"] = []
    results["P2 (naive first-K)"] = []
    results["P3 (low-activity)"] = []
    results["P3 + smooth(w=3)"] = []
    results["Generic"] = []

    for thresh in thresholds:
        for alpha in alphas:
            key = f"Online(τ={thresh},α={alpha})"
            results[key] = []

    t0 = time.time()
    for idx, (subj_id, frames) in enumerate(sorted(chrono.items())):
        aus = extract_aus(model, [f["path"] for f in frames])
        labels = (np.array([f["pspi"] for f in frames]) > 0).astype(int)
        if sum(labels) < 5 or sum(1-labels) < 5: continue

        first_k = aus[:K]
        test_aus = aus[K:]
        test_labels = labels[K:]
        if len(set(test_labels)) < 2: continue

        # P1: clean label-informed
        neutral = aus[labels == 0]
        m1, s1 = np.mean(neutral, 0), np.maximum(np.std(neutral, 0), 1e-4)
        scores_p1 = [score_baud_frame(m1, s1, f) for f in aus]
        try: results["P1 (clean baseline)"].append(roc_auc_score(labels, scores_p1))
        except: results["P1 (clean baseline)"].append(0.5)

        # P2: naive first-K
        mn, sn = np.mean(first_k, 0), np.maximum(np.std(first_k, 0), 1e-4)
        scores_p2 = [score_baud_frame(mn, sn, f) for f in test_aus]
        try: results["P2 (naive first-K)"].append(roc_auc_score(test_labels, scores_p2))
        except: results["P2 (naive first-K)"].append(0.5)

        # P3: low-activity
        lowact = get_lowact_window(first_k)
        ml, sl = np.mean(lowact, 0), np.maximum(np.std(lowact, 0), 1e-4)
        scores_p3 = [score_baud_frame(ml, sl, f) for f in test_aus]
        try: results["P3 (low-activity)"].append(roc_auc_score(test_labels, scores_p3))
        except: results["P3 (low-activity)"].append(0.5)

        # P3 + temporal smoothing
        from scipy.ndimage import uniform_filter1d
        aus_smooth = np.copy(aus)
        for j in range(aus.shape[1]):
            aus_smooth[:, j] = uniform_filter1d(aus[:, j], size=3)
        test_smooth = aus_smooth[K:]
        scores_p3s = [score_baud_frame(ml, sl, f) for f in test_smooth]
        try: results["P3 + smooth(w=3)"].append(roc_auc_score(test_labels, scores_p3s))
        except: results["P3 + smooth(w=3)"].append(0.5)

        # Generic
        gen_scores = [np.mean(f[PAIN_IDX]) for f in test_aus]
        try: results["Generic"].append(roc_auc_score(test_labels, gen_scores))
        except: results["Generic"].append(0.5)

        # Online adaptation variants
        for thresh in thresholds:
            for alpha in alphas:
                key = f"Online(τ={thresh},α={alpha})"
                # Start from low-activity baseline
                mean_online = ml.copy()
                std_online = sl.copy()
                # Track running stats for std update
                running_frames = list(lowact)

                scores_online = []
                for frame in test_aus:
                    # Score with current baseline
                    s = score_baud_frame(mean_online, std_online, frame)
                    scores_online.append(s)

                    # If model thinks this is non-pain, update baseline
                    if s < thresh:
                        # EMA update of mean
                        mean_online = (1 - alpha) * mean_online + alpha * frame
                        # Track frames for std update
                        running_frames.append(frame)
                        if len(running_frames) > 200:
                            running_frames = running_frames[-200:]
                        std_online = np.maximum(np.std(running_frames, axis=0), 1e-4)

                try: results[key].append(roc_auc_score(test_labels, scores_online))
                except: results[key].append(0.5)

        if (idx+1) % 5 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    # Print results
    generic_mean = np.mean(results["Generic"])
    print(f"\n{'='*70}")
    print(f"  ONLINE ADAPTATION RESULTS")
    print(f"  Generic: {generic_mean:.4f}")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'AUC':>8} {'vs Generic':>12} {'vs P3':>10}")
    print(f"  {'-'*62}")

    p3_mean = np.mean(results["P3 (low-activity)"])
    # Sort by AUC
    sorted_results = sorted(results.items(), key=lambda x: -np.mean(x[1]))
    for name, aucs in sorted_results:
        m = np.mean(aucs)
        marker = " ✓" if m > generic_mean else ""
        print(f"  {name:<30} {m:>8.4f} {m-generic_mean:>+12.4f} {m-p3_mean:>+10.4f}{marker}")

    print(f"{'='*70}")

    # Find best online config
    best_online = max(
        [(k, np.mean(v)) for k, v in results.items() if "Online" in k],
        key=lambda x: x[1]
    )
    print(f"\n  Best online config: {best_online[0]}, AUC={best_online[1]:.4f}")
    print(f"  Improvement over P3: {best_online[1] - p3_mean:+.4f}")
    print(f"  Improvement over P3+smooth: {best_online[1] - np.mean(results['P3 + smooth(w=3)']):+.4f}")

    return results


# ============================================================================
# EXPERIMENT 2: Per-Subject P1 vs P2 Scatter Plot
# ============================================================================

def run_per_subject_scatter(model, chrono):
    """Scatter plot of each subject's P1 AUC vs P2 AUC."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Per-Subject P1 → P2 Scatter")
    print("=" * 70)

    subj_data = []
    t0 = time.time()

    for idx, (subj_id, frames) in enumerate(sorted(chrono.items())):
        aus = extract_aus(model, [f["path"] for f in frames])
        pspis = np.array([f["pspi"] for f in frames])
        labels = (pspis > 0).astype(int)
        if sum(labels) < 5 or sum(1-labels) < 5: continue

        first_k = aus[:K]
        test_aus = aus[K:]
        test_labels = labels[K:]
        if len(set(test_labels)) < 2: continue

        # P1
        neutral = aus[labels == 0]
        m1, s1 = np.mean(neutral, 0), np.maximum(np.std(neutral, 0), 1e-4)
        scores_p1 = [score_baud_frame(m1, s1, f) for f in test_aus]

        # P2
        mn, sn = np.mean(first_k, 0), np.maximum(np.std(first_k, 0), 1e-4)
        scores_p2 = [score_baud_frame(mn, sn, f) for f in test_aus]

        try: auc_p1 = roc_auc_score(test_labels, scores_p1)
        except: auc_p1 = 0.5
        try: auc_p2 = roc_auc_score(test_labels, scores_p2)
        except: auc_p2 = 0.5

        # Pain contamination in first K
        pain_in_k = sum(1 for f in frames[:K] if f["pspi"] > 0)
        # Settling-in artifact: std of pain AUs in first K vs rest
        pain_au_std_k = np.mean(np.std(first_k[:, PAIN_IDX], axis=0))
        pain_au_std_rest = np.mean(np.std(aus[K:][:, PAIN_IDX], axis=0))
        settling_ratio = pain_au_std_k / (pain_au_std_rest + 1e-6)

        subj_data.append({
            "id": subj_id, "p1": auc_p1, "p2": auc_p2,
            "pain_in_k": pain_in_k, "settling": settling_ratio,
            "degradation": auc_p1 - auc_p2,
        })

        if (idx+1) % 10 == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    # Print per-subject breakdown
    print(f"\n  {'Subject':<10} {'P1 AUC':>8} {'P2 AUC':>8} {'Δ':>8} {'Pain/K':>8} {'Settle':>8}")
    print(f"  {'-'*55}")
    for s in sorted(subj_data, key=lambda x: -x["degradation"]):
        print(f"  {s['id']:<10} {s['p1']:>8.3f} {s['p2']:>8.3f} "
              f"{s['degradation']:>+8.3f} {s['pain_in_k']:>8} {s['settling']:>8.2f}")

    # Correlation: degradation vs pain contamination and settling
    degs = [s["degradation"] for s in subj_data]
    pains = [s["pain_in_k"] for s in subj_data]
    settles = [s["settling"] for s in subj_data]
    from scipy.stats import pearsonr
    r_pain, p_pain = pearsonr(degs, pains)
    r_settle, p_settle = pearsonr(degs, settles)
    print(f"\n  Correlation with degradation:")
    print(f"    Pain contamination: r={r_pain:.3f}, p={p_pain:.3f}")
    print(f"    Settling-in ratio:  r={r_settle:.3f}, p={p_settle:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: P1 vs P2
    p1s = [s["p1"] for s in subj_data]
    p2s = [s["p2"] for s in subj_data]
    pain_counts = [s["pain_in_k"] for s in subj_data]

    sc = ax1.scatter(p1s, p2s, c=pain_counts, cmap="YlOrRd", s=80,
                     edgecolors="black", linewidths=0.5, zorder=5)
    ax1.plot([0.4, 1], [0.4, 1], "k--", alpha=0.3, label="P1 = P2")
    ax1.set_xlabel("P1 AUC (clean baseline)", fontsize=12)
    ax1.set_ylabel("P2 AUC (naive first-K)", fontsize=12)
    ax1.set_title("Per-Subject: Clean vs Naive Baseline\n(color = pain frames in first K)",
                  fontsize=12, fontweight="bold")
    plt.colorbar(sc, ax=ax1, label="Pain frames in first K")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Scatter: degradation vs settling ratio
    ax2.scatter(settles, degs, c="#2196F3", s=80, edgecolors="black",
                linewidths=0.5, zorder=5)
    ax2.set_xlabel("Settling-in Ratio (σ_first-K / σ_rest)", fontsize=12)
    ax2.set_ylabel("AUC Degradation (P1 - P2)", fontsize=12)
    ax2.set_title(f"Degradation vs Settling-in Artifact\n(r={r_settle:.3f}, p={p_settle:.3f})",
                  fontsize=12, fontweight="bold")
    ax2.axhline(y=0, color="red", linestyle=":", alpha=0.4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/per_subject_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: per_subject_scatter.png")

    return subj_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  FINAL HIGH-IMPACT EXPERIMENTS")
    print("  1. Online Baseline Adaptation")
    print("  2. Per-Subject P1→P2 Scatter")
    print("=" * 70)

    model = load_model()
    print("  ✅ Model loaded")
    chrono = get_chronological_frames()
    print(f"  {len(chrono)} subjects")

    results = run_online_adaptation(model, chrono)
    subj_data = run_per_subject_scatter(model, chrono)

    print(f"\n{'='*70}")
    print(f"  ✅ ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"  📤 Share FULL console output + scatter plot!")


if __name__ == "__main__":
    main()
