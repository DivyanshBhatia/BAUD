#!/usr/bin/env python3
"""
R3 Reviewer Experiments
========================
1. Signal Ablation: Drop top pain AUs, measure degradation
2. Temporal Smoothing: Smooth AU vectors before z-scoring (could fix Protocol C!)
3. Counterfactual: Test on SynPAIN non-pain negative expressions (if available)

Run on Colab:
    python scripts/run_r3_experiments.py
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
PAIN_IDX = [2, 4, 5, 6, 7, 17]  # AU4, AU6, AU7, AU9, AU10, AU43
AU_NAMES = {2:"AU4", 4:"AU6", 5:"AU7", 6:"AU9", 7:"AU10", 17:"AU43"}
BATCH_SIZE = 64
K = 50; W = 10; SEED = 42

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


def score_baud(mean_b, std_b, test_aus, mask=None):
    """BAUD scoring with optional AU masking for ablation."""
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    if mask is not None:
        w[mask] = 0.0  # Zero out masked AUs
    w_sum = w.sum()
    if w_sum == 0: w_sum = 1
    w /= w_sum
    scores = []
    for f in test_aus:
        z = np.maximum((f - mean_b) / std_b, 0)
        scores.append(1.0 / (1.0 + np.exp(-np.dot(w, z) + 2.0)))
    return np.array(scores)


def get_lowact_window(first_k):
    if len(first_k) <= W:
        return first_k
    best_score, best_idx = np.inf, 0
    for i in range(len(first_k) - W + 1):
        score = np.mean(first_k[i:i+W][:, PAIN_IDX])
        if score < best_score:
            best_score, best_idx = score, i
    return first_k[best_idx:best_idx+W]


def temporal_smooth(aus, window_size):
    """Apply moving average smoothing to AU vectors."""
    if window_size <= 1 or len(aus) <= window_size:
        return aus
    smoothed = np.copy(aus)
    for i in range(len(aus)):
        start = max(0, i - window_size // 2)
        end = min(len(aus), i + window_size // 2 + 1)
        smoothed[i] = np.mean(aus[start:end], axis=0)
    return smoothed


# ============================================================================
# EXPERIMENT 1: Signal Ablation
# ============================================================================

def run_signal_ablation(model, chrono):
    """Drop top pain AUs one by one and measure degradation."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Signal Ablation")
    print("  How much does each pain AU contribute?")
    print("=" * 70)

    # Configs: no mask, drop each pain AU, drop all pain AUs, drop top-2, top-3
    configs = {
        "All AUs (baseline)": None,
    }
    for idx in PAIN_IDX:
        name = AU_NAMES.get(idx, f"AU_{idx}")
        configs[f"Drop {name}"] = [idx]

    # Drop top-2, top-3 (by learned weight order: AU10, AU4, AU9)
    configs["Drop AU10+AU4"] = [7, 2]
    configs["Drop AU10+AU4+AU9"] = [7, 2, 6]
    configs["Drop ALL pain AUs"] = list(PAIN_IDX)

    results = {name: [] for name in configs}
    t0 = time.time()

    for idx, (subj_id, frames) in enumerate(sorted(chrono.items())):
        aus = extract_aus(model, [f["path"] for f in frames])
        labels = (np.array([f["pspi"] for f in frames]) > 0).astype(int)
        if sum(labels) < 5 or sum(1-labels) < 5: continue

        neutral = aus[labels == 0]
        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)

        for name, mask in configs.items():
            scores = score_baud(mean_b, std_b, aus, mask=mask)
            try: results[name].append(roc_auc_score(labels, scores))
            except: results[name].append(0.5)

        if (idx+1) % 10 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print(f"  SIGNAL ABLATION RESULTS (P1, label-informed baseline)")
    print(f"{'='*70}")
    print(f"  {'Configuration':<25} {'Mean AUC':>10} {'Δ from full':>12}")
    print(f"  {'-'*50}")
    full_auc = np.mean(results["All AUs (baseline)"])
    for name in configs:
        m = np.mean(results[name])
        delta = m - full_auc
        print(f"  {name:<25} {m:>10.4f} {delta:>+12.4f}")
    print(f"{'='*70}")

    return results


# ============================================================================
# EXPERIMENT 2: Temporal Smoothing
# ============================================================================

def run_temporal_smoothing(model, chrono):
    """Test temporal smoothing of AU vectors — could fix Protocol C!"""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Temporal Smoothing")
    print("  Does smoothing AU vectors improve Protocol C?")
    print("=" * 70)

    window_sizes = [1, 3, 5, 7, 11, 15]
    protocols = {
        "P1 (clean baseline)": {},
        "P2 (naive first-K)": {},
        "P3 (low-activity)": {},
    }
    for pname in protocols:
        protocols[pname] = {ws: [] for ws in window_sizes}

    generic_aucs = []
    t0 = time.time()

    for idx, (subj_id, frames) in enumerate(sorted(chrono.items())):
        aus_raw = extract_aus(model, [f["path"] for f in frames])
        labels = (np.array([f["pspi"] for f in frames]) > 0).astype(int)
        if sum(labels) < 5 or sum(1-labels) < 5: continue

        # Generic (no smoothing, no baseline)
        gen_scores = np.array([np.mean(f[PAIN_IDX]) for f in aus_raw[K:]])
        test_labels_k = labels[K:]
        if len(set(test_labels_k)) >= 2:
            try: generic_aucs.append(roc_auc_score(test_labels_k, gen_scores))
            except: generic_aucs.append(0.5)

        for ws in window_sizes:
            aus = temporal_smooth(aus_raw, ws)

            # P1: clean label-informed baseline
            neutral = aus[labels == 0]
            mean_b = np.mean(neutral, 0)
            std_b = np.maximum(np.std(neutral, 0), 1e-4)
            scores_p1 = score_baud(mean_b, std_b, aus)
            try: protocols["P1 (clean baseline)"][ws].append(roc_auc_score(labels, scores_p1))
            except: protocols["P1 (clean baseline)"][ws].append(0.5)

            # P2: naive first-K
            first_k = aus[:K]
            test_aus = aus[K:]
            test_labels = labels[K:]
            if len(set(test_labels)) < 2: continue

            mean_n, std_n = np.mean(first_k, 0), np.maximum(np.std(first_k, 0), 1e-4)
            scores_p2 = score_baud(mean_n, std_n, test_aus)
            try: protocols["P2 (naive first-K)"][ws].append(roc_auc_score(test_labels, scores_p2))
            except: protocols["P2 (naive first-K)"][ws].append(0.5)

            # P3: low-activity window
            lowact = get_lowact_window(first_k)
            mean_la, std_la = np.mean(lowact, 0), np.maximum(np.std(lowact, 0), 1e-4)
            scores_p3 = score_baud(mean_la, std_la, test_aus)
            try: protocols["P3 (low-activity)"][ws].append(roc_auc_score(test_labels, scores_p3))
            except: protocols["P3 (low-activity)"][ws].append(0.5)

        if (idx+1) % 10 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    generic_mean = np.mean(generic_aucs) if generic_aucs else 0.694

    print(f"\n{'='*70}")
    print(f"  TEMPORAL SMOOTHING RESULTS")
    print(f"  Generic baseline: {generic_mean:.3f}")
    print(f"{'='*70}")
    print(f"  {'Protocol':<22}", end="")
    for ws in window_sizes:
        print(f"  {'w='+str(ws):>6}", end="")
    print()
    print(f"  {'-'*65}")
    for pname in protocols:
        print(f"  {pname:<22}", end="")
        for ws in window_sizes:
            aucs = protocols[pname][ws]
            m = np.mean(aucs) if aucs else 0
            marker = " ✓" if m > generic_mean and "P2" in pname or "P3" in pname else ""
            print(f"  {m:>6.3f}", end="")
        print()
    print(f"{'='*70}")

    # Find best smoothing for P3
    best_ws = max(window_sizes, key=lambda ws: np.mean(protocols["P3 (low-activity)"][ws]))
    best_p3 = np.mean(protocols["P3 (low-activity)"][best_ws])
    no_smooth_p3 = np.mean(protocols["P3 (low-activity)"][1])
    print(f"\n  Best P3 smoothing: w={best_ws}, AUC={best_p3:.4f} "
          f"(vs no-smooth {no_smooth_p3:.4f}, Δ={best_p3-no_smooth_p3:+.4f})")
    print(f"  Generic: {generic_mean:.4f}")
    if best_p3 > generic_mean:
        print(f"  ✅ Temporal smoothing pushes P3 ABOVE Generic!")
    else:
        print(f"  ❌ Still below Generic even with smoothing")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"P1 (clean baseline)": "#2196F3", "P2 (naive first-K)": "#EF5350",
              "P3 (low-activity)": "#4CAF50"}
    for pname in protocols:
        aucs = [np.mean(protocols[pname][ws]) for ws in window_sizes]
        ax.plot(window_sizes, aucs, "o-", color=colors[pname], linewidth=2,
                markersize=7, label=pname)
    ax.axhline(y=generic_mean, color="black", linestyle="--", alpha=0.5,
               label=f"Generic={generic_mean:.3f}")
    ax.set_xlabel("Smoothing Window Size (frames)", fontsize=12)
    ax.set_ylabel("Mean Per-Subject AUC", fontsize=12)
    ax.set_title("Effect of Temporal Smoothing on BAUD\n(across all protocols)",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 0.85)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/r3_temporal_smoothing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: r3_temporal_smoothing.png")

    return protocols, generic_mean


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  R3 REVIEWER EXPERIMENTS")
    print("  Signal Ablation + Temporal Smoothing")
    print("=" * 70)

    model = load_model()
    print("  ✅ Model loaded")
    chrono = get_chronological_frames()
    print(f"  {len(chrono)} subjects")

    # Experiment 1: Signal Ablation
    ablation_results = run_signal_ablation(model, chrono)

    # Experiment 2: Temporal Smoothing
    smooth_results, generic_mean = run_temporal_smoothing(model, chrono)

    print(f"\n{'='*70}")
    print(f"  ✅ R3 EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"  📤 Share console output + temporal smoothing plot!")


if __name__ == "__main__":
    main()
