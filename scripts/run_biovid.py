#!/usr/bin/env python3
"""
BAUD × BioVid Heat Pain Database (Part A)
==========================================
87 subjects, 5 conditions (BL, PA1-PA4), 20 reps each, 5.5s videos.
Tests binary AND graded pain detection with three-protocol evaluation.

Run on Colab:
    python scripts/run_biovid.py
"""
import os, sys, glob, time, cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict, defaultdict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO_DIR = "/content/biovid_video/video"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
BATCH_SIZE = 64
FRAMES_PER_VIDEO = 5  # Sample 5 frames from middle of each 5.5s video
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


def extract_frames_from_video(video_path, n_frames=5):
    """Extract n_frames evenly spaced from the middle of the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        n_frames = max(1, total)

    # Sample from middle 60% of video (avoid start/end artifacts)
    start = int(total * 0.2)
    end = int(total * 0.8)
    indices = np.linspace(start, end, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def extract_aus_from_frames(model, frames):
    """Extract AU vectors from PIL Image frames."""
    if not frames:
        return np.array([])
    tensors = [au_transform(f) for f in frames]
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        out = model(batch)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return torch.sigmoid(pred).cpu().numpy()


def parse_filename(filename):
    """Parse BioVid filename: {subject_id}-{condition}-{trial}.mp4"""
    name = os.path.splitext(filename)[0]
    parts = name.split("-")
    if len(parts) >= 3:
        subject = parts[0]
        condition = parts[1]  # BL1, PA1, PA2, PA3, PA4
        trial = parts[2]
        # Map condition to pain level
        level_map = {"BL1": 0, "PA1": 1, "PA2": 2, "PA3": 3, "PA4": 4}
        level = level_map.get(condition, -1)
        return subject, condition, level, trial
    return None, None, -1, None


def score_baud(mean_b, std_b, test_aus):
    w = np.ones(test_aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()
    scores = []
    for f in test_aus:
        z = np.maximum((f - mean_b) / std_b, 0)
        scores.append(1.0 / (1.0 + np.exp(-np.dot(w, z) + 2.0)))
    return np.array(scores)


def main():
    print("=" * 70)
    print("  BAUD × BioVid Heat Pain Database (Part A)")
    print("  87 subjects, 5 conditions, graded pain intensity")
    print("=" * 70)

    model = load_model()
    print(f"  ✅ Model on {DEVICE}")

    # ── Step 1: Parse all videos ──
    print("\n📂 Parsing video files...")
    subjects = defaultdict(lambda: defaultdict(list))
    for subj_folder in sorted(os.listdir(VIDEO_DIR)):
        subj_path = os.path.join(VIDEO_DIR, subj_folder)
        if not os.path.isdir(subj_path): continue
        for vid_file in sorted(os.listdir(subj_path)):
            if not vid_file.endswith(('.mp4', '.avi')): continue
            subj_id, condition, level, trial = parse_filename(vid_file)
            if subj_id and level >= 0:
                subjects[subj_id][level].append(os.path.join(subj_path, vid_file))

    print(f"  Subjects: {len(subjects)}")
    # Show distribution
    for subj in sorted(subjects.keys())[:3]:
        dist = {f"L{k}": len(v) for k, v in sorted(subjects[subj].items())}
        print(f"    {subj}: {dist}")

    # ── Step 2: Extract AUs ──
    print(f"\n🔍 Extracting AUs ({FRAMES_PER_VIDEO} frames/video)...")
    au_data = {}  # {subj: {level: np.array of shape (n_videos*n_frames, 41)}}
    t0 = time.time()
    total_frames = 0

    for idx, (subj_id, levels) in enumerate(sorted(subjects.items())):
        au_data[subj_id] = {}
        for level, video_paths in sorted(levels.items()):
            all_aus = []
            for vpath in video_paths:
                frames = extract_frames_from_video(vpath, FRAMES_PER_VIDEO)
                if frames:
                    aus = extract_aus_from_frames(model, frames)
                    if len(aus) > 0:
                        all_aus.append(aus)
            if all_aus:
                au_data[subj_id][level] = np.concatenate(all_aus)
                total_frames += len(au_data[subj_id][level])

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Subject {idx+1}/{len(subjects)} "
                  f"({total_frames} frames, {elapsed:.0f}s)")

    print(f"  ✅ {total_frames} frames from {len(au_data)} subjects "
          f"in {time.time()-t0:.0f}s")

    # Cache
    cache = {}
    for s, levels in au_data.items():
        for l, aus in levels.items():
            cache[f"{s}_L{l}"] = aus
    np.savez(f"{RESULTS_DIR}/biovid_aus.npz", **cache)
    print(f"  💾 Cached to biovid_aus.npz")

    # ── Step 3: Binary Pain Detection (BL vs PA4) ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1: Binary Pain Detection (BL vs PA4)")
    print(f"{'='*70}")

    methods = {"BAUD": [], "Mahalanobis": [], "Generic": []}

    for subj_id, levels in sorted(au_data.items()):
        if 0 not in levels or 4 not in levels: continue
        neutral = levels[0]
        pain = levels[4]

        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)

        all_frames = np.concatenate([neutral, pain])
        all_labels = np.array([0]*len(neutral) + [1]*len(pain))

        # BAUD
        baud_s = score_baud(mean_b, std_b, all_frames)
        try: methods["BAUD"].append(roc_auc_score(all_labels, baud_s))
        except: methods["BAUD"].append(0.5)

        # Mahalanobis
        fb = neutral[:, PAIN_IDX]
        mm, mc = np.mean(fb, 0), np.cov(fb, rowvar=False) + np.eye(len(PAIN_IDX))*1e-4
        mi = np.linalg.inv(mc)
        mah_s = [1.0/(1.0+np.exp(-np.sqrt((f-mm)@mi@(f-mm))+3.0))
                 for f in all_frames[:, PAIN_IDX]]
        try: methods["Mahalanobis"].append(roc_auc_score(all_labels, mah_s))
        except: methods["Mahalanobis"].append(0.5)

        # Generic
        gen_s = [np.mean(f[PAIN_IDX]) for f in all_frames]
        try: methods["Generic"].append(roc_auc_score(all_labels, gen_s))
        except: methods["Generic"].append(0.5)

    print(f"  {'Method':<20} {'Mean AUC':>10} {'Std':>8} {'Subjects':>10}")
    print(f"  {'-'*50}")
    for name, aucs in methods.items():
        print(f"  {name:<20} {np.mean(aucs):>10.4f} {np.std(aucs):>8.4f} {len(aucs):>10}")

    # ── Step 4: Graded Pain Detection (BL vs each level) ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 2: Graded Pain Detection (BL vs PA1/PA2/PA3/PA4)")
    print(f"{'='*70}")

    for pain_level in [1, 2, 3, 4]:
        baud_aucs, gen_aucs = [], []
        for subj_id, levels in sorted(au_data.items()):
            if 0 not in levels or pain_level not in levels: continue
            neutral, pain = levels[0], levels[pain_level]
            mean_b = np.mean(neutral, 0)
            std_b = np.maximum(np.std(neutral, 0), 1e-4)

            all_f = np.concatenate([neutral, pain])
            all_l = np.array([0]*len(neutral) + [1]*len(pain))

            baud_s = score_baud(mean_b, std_b, all_f)
            gen_s = [np.mean(f[PAIN_IDX]) for f in all_f]
            try:
                baud_aucs.append(roc_auc_score(all_l, baud_s))
                gen_aucs.append(roc_auc_score(all_l, gen_s))
            except: pass

        gap = np.mean(baud_aucs) - np.mean(gen_aucs)
        print(f"  BL vs PA{pain_level}: BAUD={np.mean(baud_aucs):.4f}±{np.std(baud_aucs):.4f}  "
              f"Generic={np.mean(gen_aucs):.4f}  Gap={gap:+.4f}")

    # ── Step 5: Intensity Discrimination ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 3: Intensity Discrimination")
    print(f"  Can BAUD's score distinguish PA1 < PA2 < PA3 < PA4?")
    print(f"{'='*70}")

    baud_by_level = defaultdict(list)
    gen_by_level = defaultdict(list)

    for subj_id, levels in sorted(au_data.items()):
        if 0 not in levels: continue
        neutral = levels[0]
        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)

        for level in range(5):
            if level not in levels: continue
            aus = levels[level]
            baud_s = score_baud(mean_b, std_b, aus)
            gen_s = [np.mean(f[PAIN_IDX]) for f in aus]
            baud_by_level[level].append(float(np.mean(baud_s)))
            gen_by_level[level].append(float(np.mean(gen_s)))

    print(f"  {'Level':<10} {'BAUD score':>12} {'Generic score':>14}")
    print(f"  {'-'*38}")
    for level in range(5):
        label = "BL" if level == 0 else f"PA{level}"
        b = np.mean(baud_by_level[level])
        g = np.mean(gen_by_level[level])
        print(f"  {label:<10} {b:>12.4f} {g:>14.4f}")

    # Spearman correlation between level and score
    all_levels_b, all_scores_b = [], []
    all_levels_g, all_scores_g = [], []
    for level in range(5):
        for s in baud_by_level[level]:
            all_levels_b.append(level); all_scores_b.append(s)
        for s in gen_by_level[level]:
            all_levels_g.append(level); all_scores_g.append(s)
    r_baud, p_baud = spearmanr(all_levels_b, all_scores_b)
    r_gen, p_gen = spearmanr(all_levels_g, all_scores_g)
    print(f"\n  Intensity correlation:")
    print(f"    BAUD:    Spearman r={r_baud:.4f}, p={p_baud:.2e}")
    print(f"    Generic: Spearman r={r_gen:.4f}, p={p_gen:.2e}")

    # ── Step 6: Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graded detection AUC
    levels_plot = [1, 2, 3, 4]
    baud_graded, gen_graded = [], []
    for pl in levels_plot:
        b_aucs, g_aucs = [], []
        for subj_id, levels in sorted(au_data.items()):
            if 0 not in levels or pl not in levels: continue
            neutral, pain = levels[0], levels[pl]
            mean_b = np.mean(neutral, 0)
            std_b = np.maximum(np.std(neutral, 0), 1e-4)
            all_f = np.concatenate([neutral, pain])
            all_l = np.array([0]*len(neutral) + [1]*len(pain))
            try:
                b_aucs.append(roc_auc_score(all_l, score_baud(mean_b, std_b, all_f)))
                g_aucs.append(roc_auc_score(all_l, [np.mean(f[PAIN_IDX]) for f in all_f]))
            except: pass
        baud_graded.append(np.mean(b_aucs))
        gen_graded.append(np.mean(g_aucs))

    x = np.arange(len(levels_plot))
    ax1.bar(x - 0.175, baud_graded, 0.35, label="BAUD", color="#2196F3", alpha=0.8)
    ax1.bar(x + 0.175, gen_graded, 0.35, label="Generic", color="#9E9E9E", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"BL vs PA{l}" for l in levels_plot])
    ax1.set_ylabel("Mean Per-Subject AUC")
    ax1.set_title("BioVid: Graded Pain Detection", fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y"); ax1.set_ylim(0.4, 1.0)
    for i, (b, g) in enumerate(zip(baud_graded, gen_graded)):
        ax1.text(i-0.175, b+0.02, f"{b:.3f}", ha="center", fontsize=9)
        ax1.text(i+0.175, g+0.02, f"{g:.3f}", ha="center", fontsize=9)

    # Intensity discrimination
    levels_all = [0, 1, 2, 3, 4]
    baud_means = [np.mean(baud_by_level[l]) for l in levels_all]
    gen_means = [np.mean(gen_by_level[l]) for l in levels_all]
    ax2.plot(levels_all, baud_means, "o-", color="#2196F3", linewidth=2,
             markersize=8, label=f"BAUD (ρ={r_baud:.3f})")
    ax2.plot(levels_all, gen_means, "s--", color="#9E9E9E", linewidth=1.5,
             markersize=7, label=f"Generic (ρ={r_gen:.3f})")
    ax2.set_xticks(levels_all)
    ax2.set_xticklabels(["BL", "PA1", "PA2", "PA3", "PA4"])
    ax2.set_xlabel("Pain Level"); ax2.set_ylabel("Mean Pain Score")
    ax2.set_title("BioVid: Intensity Discrimination", fontsize=13, fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/biovid_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: biovid_results.png")

    # Summary
    print(f"\n{'='*70}")
    print(f"  BIOVID SUMMARY")
    print(f"{'='*70}")
    print(f"  Binary (BL vs PA4):     BAUD={np.mean(methods['BAUD']):.4f}, "
          f"Generic={np.mean(methods['Generic']):.4f}")
    print(f"  Intensity correlation:  BAUD ρ={r_baud:.4f}, Generic ρ={r_gen:.4f}")
    print(f"  Personalization gap:    {np.mean(methods['BAUD'])-np.mean(methods['Generic']):+.4f}")
    print(f"{'='*70}")
    print(f"  📤 Share console output + plot!")


if __name__ == "__main__":
    main()
