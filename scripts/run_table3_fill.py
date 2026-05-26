#!/usr/bin/env python3
"""
Fill Table 3: Low-activity heuristic for ALL personalized methods.
Quick script — uses cached chronological AUs from run_protocol_c.py
or re-extracts if needed.

Run on Colab:
    python scripts/run_table3_fill.py
"""
import os, sys, time, numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

IMAGES_DIR = "/content/Images/Images"
LABELS_DIR = "/content/Frame_Labels/Frame_Labels/PSPI"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
RESULTS_DIR = "/content/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
BATCH_SIZE = 64
W = 10  # Low-activity window size (fixed on validation)
K = 50  # First K chronological frames

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


# ── Baseline estimation ──

def get_naive_baseline(first_k):
    """Use all first-K frames as-is."""
    return np.mean(first_k, 0), np.maximum(np.std(first_k, 0), 1e-4)

def get_lowact_baseline(first_k):
    """Low-activity window: select W frames with lowest mean pain-AU activation."""
    if len(first_k) <= W:
        return get_naive_baseline(first_k)
    best_score, best_idx = np.inf, 0
    for i in range(len(first_k) - W + 1):
        score = np.mean(first_k[i:i+W][:, PAIN_IDX])
        if score < best_score:
            best_score, best_idx = score, i
    selected = first_k[best_idx:best_idx+W]
    return np.mean(selected, 0), np.maximum(np.std(selected, 0), 1e-4)

def get_full_neutral_baseline(aus, labels):
    """Label-informed: use all neutral frames."""
    neutral = aus[labels == 0]
    return np.mean(neutral, 0), np.maximum(np.std(neutral, 0), 1e-4)


# ── Scoring methods ──

def score_baud(mean_b, std_b, test_aus):
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()
    scores = []
    for f in test_aus:
        z = np.maximum((f - mean_b) / std_b, 0)
        scores.append(1.0 / (1.0 + np.exp(-np.dot(w, z) + 2.0)))
    return np.array(scores)

def score_median_mad(mean_b, std_b, test_aus):
    """Median/MAD uses the same baseline frames but computes median+MAD."""
    # Note: mean_b/std_b are from the SELECTED frames (naive or lowact)
    # For Median/MAD, we recompute using median and MAD from same frames
    # We need the raw frames, not just mean/std
    # → handled in the main loop below
    pass

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

def score_median_mad_from_frames(baseline_frames, test_aus):
    median_b = np.median(baseline_frames, 0)
    mad_b = np.maximum(np.median(np.abs(baseline_frames - median_b), axis=0), 1e-4)
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()
    scores = []
    for f in test_aus:
        z = np.maximum((f - median_b) / (1.4826 * mad_b), 0)
        scores.append(1.0 / (1.0 + np.exp(-np.dot(w, z) + 2.0)))
    return np.array(scores)


def get_lowact_frames(first_k):
    """Return the actual low-activity window frames."""
    if len(first_k) <= W:
        return first_k
    best_score, best_idx = np.inf, 0
    for i in range(len(first_k) - W + 1):
        score = np.mean(first_k[i:i+W][:, PAIN_IDX])
        if score < best_score:
            best_score, best_idx = score, i
    return first_k[best_idx:best_idx+W]


# ── Main ──

def main():
    print("=" * 70)
    print("  TABLE 3: Low-Activity Heuristic for ALL Methods")
    print("=" * 70)

    model = load_model()
    print("  ✅ Model loaded")
    chrono = get_chronological_frames()
    print(f"  {len(chrono)} subjects\n")

    # Methods × Protocols
    configs = {
        "BAUD":        {"full": [], "naive": [], "lowact": []},
        "Median/MAD":  {"full": [], "naive": [], "lowact": []},
        "Mahalanobis": {"full": [], "naive": [], "lowact": []},
        "Generic":     {"full": [], "naive": [], "lowact": []},
    }

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

        # Get baseline frames for each protocol
        naive_frames = first_k
        lowact_frames = get_lowact_frames(first_k)
        neutral_frames = aus[labels == 0]

        # Naive baselines
        naive_mean, naive_std = get_naive_baseline(first_k)
        lowact_mean, lowact_std = get_lowact_baseline(first_k)
        full_mean, full_std = get_full_neutral_baseline(aus, labels)

        def safe_auc(labels, scores):
            try: return roc_auc_score(labels, scores)
            except: return 0.5

        # BAUD
        configs["BAUD"]["full"].append(safe_auc(test_labels, score_baud(full_mean, full_std, test_aus)))
        configs["BAUD"]["naive"].append(safe_auc(test_labels, score_baud(naive_mean, naive_std, test_aus)))
        configs["BAUD"]["lowact"].append(safe_auc(test_labels, score_baud(lowact_mean, lowact_std, test_aus)))

        # Median/MAD
        configs["Median/MAD"]["full"].append(safe_auc(test_labels, score_median_mad_from_frames(neutral_frames, test_aus)))
        configs["Median/MAD"]["naive"].append(safe_auc(test_labels, score_median_mad_from_frames(naive_frames, test_aus)))
        configs["Median/MAD"]["lowact"].append(safe_auc(test_labels, score_median_mad_from_frames(lowact_frames, test_aus)))

        # Mahalanobis
        configs["Mahalanobis"]["full"].append(safe_auc(test_labels, score_mahalanobis(neutral_frames, test_aus)))
        configs["Mahalanobis"]["naive"].append(safe_auc(test_labels, score_mahalanobis(naive_frames, test_aus)))
        configs["Mahalanobis"]["lowact"].append(safe_auc(test_labels, score_mahalanobis(lowact_frames, test_aus)))

        # Generic (no baseline needed)
        gen_auc = safe_auc(test_labels, score_generic(test_aus))
        configs["Generic"]["full"].append(gen_auc)
        configs["Generic"]["naive"].append(gen_auc)
        configs["Generic"]["lowact"].append(gen_auc)

        if (idx+1) % 5 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(chrono)} ({time.time()-t0:.1f}s)")

    # ── Print Table 3 ──
    print(f"\n{'='*70}")
    print(f"  TABLE 3 VALUES (copy into paper)")
    print(f"{'='*70}")
    print(f"  {'Method':<15} {'Full (ref.)':>12} {'Naive K=50':>12} {'Low-act.':>12}")
    print(f"  {'-'*55}")
    for name in ["BAUD", "Median/MAD", "Mahalanobis", "Generic"]:
        f = np.mean(configs[name]["full"])
        n = np.mean(configs[name]["naive"])
        l = np.mean(configs[name]["lowact"])
        print(f"  {name:<15} {f:>12.3f} {n:>12.3f} {l:>12.3f}")
    print(f"{'='*70}")

    # Does heuristic help each method?
    print(f"\n  Per-method improvement from low-activity heuristic:")
    for name in ["BAUD", "Median/MAD", "Mahalanobis"]:
        naive_m = np.mean(configs[name]["naive"])
        lowact_m = np.mean(configs[name]["lowact"])
        generic_m = np.mean(configs["Generic"]["naive"])
        improved = sum(1 for l, n in zip(configs[name]["lowact"], configs[name]["naive"]) if l > n)
        total = len(configs[name]["naive"])
        print(f"  {name:<15}: {naive_m:.3f} → {lowact_m:.3f} "
              f"({lowact_m-naive_m:+.3f}), "
              f"{'above' if lowact_m > generic_m else 'below'} Generic ({generic_m:.3f}), "
              f"improved {improved}/{total} subjects")

    print(f"\n  📋 Copy these values into Table 3 of the paper!")


if __name__ == "__main__":
    main()
