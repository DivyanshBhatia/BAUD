#!/usr/bin/env python3
"""
Case Study Figure: BAUD in action on a real UNBC patient.
Shows raw AUs → baseline calibration → z-scores → pain scores
alongside ground-truth PSPI over time.

Run on Colab:
    python scripts/run_case_study.py
"""
import os, sys, numpy as np, torch, time
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

IMAGES_DIR = "/content/Images/Images"
LABELS_DIR = "/content/Frame_Labels/Frame_Labels/PSPI"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
RESULTS_DIR = "/content/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
BATCH_SIZE = 64

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

def get_subject_sequences(subj_id):
    """Get frames for ONE subject, organized by sequence."""
    sequences = {}
    for subj_folder in os.listdir(IMAGES_DIR):
        if not subj_folder.startswith(subj_id):
            continue
        subj_img = os.path.join(IMAGES_DIR, subj_folder)
        subj_lbl = os.path.join(LABELS_DIR, subj_folder)
        for seq in sorted(os.listdir(subj_img)):
            seq_img = os.path.join(subj_img, seq)
            seq_lbl = os.path.join(subj_lbl, seq)
            if not os.path.isdir(seq_img) or not os.path.isdir(seq_lbl):
                continue
            frames = []
            for img_file in sorted(os.listdir(seq_img)):
                if not img_file.endswith(".png"): continue
                lbl_file = os.path.join(seq_lbl, img_file.replace(".png","")+"_facs.txt")
                if os.path.exists(lbl_file):
                    try:
                        with open(lbl_file) as f: pspi = float(f.read().strip())
                        frames.append({"path": os.path.join(seq_img, img_file), "pspi": pspi})
                    except: pass
            if frames:
                sequences[seq] = frames
    return sequences


def find_best_subject():
    """Find a subject with clear pain episodes for the case study."""
    print("  Finding best subject for case study...")
    best_subj, best_score = None, 0

    for subj_folder in sorted(os.listdir(IMAGES_DIR)):
        subj_id = subj_folder.split("-")[0]
        subj_lbl = os.path.join(LABELS_DIR, subj_folder)
        if not os.path.isdir(subj_lbl): continue

        for seq in sorted(os.listdir(subj_lbl)):
            seq_lbl = os.path.join(subj_lbl, seq)
            if not os.path.isdir(seq_lbl): continue

            pspis = []
            for f in sorted(os.listdir(seq_lbl)):
                try:
                    with open(os.path.join(seq_lbl, f)) as fh:
                        pspis.append(float(fh.read().strip()))
                except: pass

            if len(pspis) < 50: continue
            n_pain = sum(1 for p in pspis if p > 0)
            max_pspi = max(pspis)
            # Good subject: has pain episodes, not too many, clear transitions
            pain_ratio = n_pain / len(pspis)
            if 0.05 < pain_ratio < 0.4 and max_pspi >= 4:
                score = max_pspi * (1 - abs(pain_ratio - 0.2))
                if score > best_score:
                    best_score = score
                    best_subj = (subj_id, seq, len(pspis), n_pain, max_pspi)

    if best_subj:
        print(f"  Selected: Subject {best_subj[0]}, seq {best_subj[1]}")
        print(f"    {best_subj[2]} frames, {best_subj[3]} pain, max PSPI={best_subj[4]}")
    return best_subj


def main():
    print("=" * 65)
    print("  CASE STUDY: BAUD in Action on a Real Patient")
    print("=" * 65)

    model = load_model()
    print("  ✅ Model loaded")

    # Find best subject
    best = find_best_subject()
    if not best:
        print("  ❌ No suitable subject found")
        return

    subj_id, seq_name = best[0], best[1]

    # Get ALL sequences for this subject
    sequences = get_subject_sequences(subj_id)
    print(f"  Subject {subj_id}: {len(sequences)} sequences")

    # Use the selected sequence
    frames = sequences[seq_name]
    print(f"  Sequence: {seq_name}, {len(frames)} frames")

    # Extract AUs
    print("  Extracting AUs...")
    aus = extract_aus(model, [f["path"] for f in frames])
    pspis = np.array([f["pspi"] for f in frames])

    # Use first 20 neutral frames as baseline (simulating calibration)
    neutral_mask = pspis == 0
    neutral_aus = aus[neutral_mask][:20]  # First 20 neutral frames
    mean_b = np.mean(neutral_aus, 0)
    std_b = np.maximum(np.std(neutral_aus, 0), 1e-4)

    # Compute BAUD scores
    w = np.ones(aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()

    baud_scores = []
    z_scores_pain = []
    for frame_aus in aus:
        z = np.maximum((frame_aus - mean_b) / std_b, 0)
        raw = np.dot(w, z)
        baud_scores.append(1.0 / (1.0 + np.exp(-raw + 2.0)))
        z_scores_pain.append(z[PAIN_IDX])

    baud_scores = np.array(baud_scores)
    z_scores_pain = np.array(z_scores_pain)

    # Generic scores for comparison
    generic_scores = np.array([np.mean(f[PAIN_IDX]) for f in aus])
    # Normalize generic to 0-1 for visual comparison
    generic_norm = (generic_scores - generic_scores.min()) / (generic_scores.max() - generic_scores.min() + 1e-8)

    # ── Generate Figure ──
    print("  Generating case study figure...")
    time_axis = np.arange(len(frames))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1.5, 1.2, 1.2], hspace=0.35)

    # Panel A: Ground truth PSPI over time
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(time_axis, pspis, alpha=0.3, color="#EF5350")
    ax1.plot(time_axis, pspis, color="#EF5350", linewidth=1.5, label="PSPI")
    ax1.set_ylabel("PSPI Score", fontsize=11)
    ax1.set_title(f"Case Study: BAUD on Subject {subj_id} (sequence {seq_name}, {len(frames)} frames)",
                  fontsize=14, fontweight="bold")
    ax1.set_xlim(0, len(frames))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2)
    # Shade pain regions
    pain_regions = pspis > 0
    ax1.fill_between(time_axis, 0, ax1.get_ylim()[1], where=pain_regions,
                     alpha=0.08, color="red", label="_nolegend_")

    # Panel B: Pain AU z-scores heatmap
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(z_scores_pain.T, aspect="auto", cmap="YlOrRd",
                     extent=[0, len(frames), -0.5, len(PAIN_IDX)-0.5],
                     vmin=0, vmax=np.percentile(z_scores_pain, 95))
    ax2.set_yticks(range(len(PAIN_NAMES)))
    ax2.set_yticklabels(PAIN_NAMES, fontsize=10)
    ax2.set_ylabel("Pain AU\nZ-Scores", fontsize=11)
    plt.colorbar(im, ax=ax2, label="Z-score deviation", shrink=0.8)
    ax2.set_xlim(0, len(frames))

    # Panel C: BAUD score vs Generic score
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_axis, baud_scores, color="#2196F3", linewidth=1.5,
             label="BAUD (personalized)", zorder=3)
    ax3.plot(time_axis, generic_norm, color="#9E9E9E", linewidth=1,
             alpha=0.6, label="Generic (normalized)", zorder=2)
    ax3.axhline(y=0.5, color="black", linestyle=":", alpha=0.3)
    ax3.set_ylabel("Pain Score", fontsize=11)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim(0, len(frames))
    ax3.fill_between(time_axis, 0, 1, where=pain_regions,
                     alpha=0.08, color="red")

    # Panel D: Top contributing AUs for peak pain frame
    ax4 = fig.add_subplot(gs[3])
    peak_idx = np.argmax(pspis)
    peak_z = z_scores_pain[peak_idx]
    contributions = peak_z * np.array([w[i] for i in PAIN_IDX])
    sort_idx = np.argsort(contributions)[::-1]

    colors_bar = ["#2196F3" if contributions[i] > 0.01 else "#90CAF9"
                  for i in sort_idx]
    ax4.barh(range(len(PAIN_NAMES)),
             [contributions[i] for i in sort_idx],
             color=[colors_bar[j] for j in range(len(sort_idx))],
             edgecolor="white", height=0.6)
    ax4.set_yticks(range(len(PAIN_NAMES)))
    ax4.set_yticklabels([PAIN_NAMES[i] for i in sort_idx], fontsize=10)
    ax4.set_xlabel("Weighted Z-Score Contribution", fontsize=11)
    ax4.set_title(f"Peak Pain Frame (#{peak_idx}, PSPI={pspis[peak_idx]:.0f}): "
                  f"Per-AU Contributions",
                  fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.2, axis="x")
    ax4.invert_yaxis()

    # Add annotations
    ax1.annotate("Pain episodes", xy=(peak_idx, pspis[peak_idx]),
                 xytext=(peak_idx + 20, pspis[peak_idx] + 1),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                 fontsize=9, color="red", fontweight="bold")

    # Calibration period marker
    cal_end = np.where(neutral_mask)[0][19] if np.sum(neutral_mask) >= 20 else 20
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(0, cal_end, alpha=0.1, color="green")
    ax1.text(cal_end/2, ax1.get_ylim()[1]*0.85, "Calibration\nperiod",
             ha="center", fontsize=8, color="green", fontweight="bold")

    plt.savefig(f"{RESULTS_DIR}/case_study.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  ✅ Saved: case_study.png")

    # Also generate a compact version for the paper (2-panel)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                          gridspec_kw={"hspace": 0.15})

    # Top: PSPI ground truth
    ax_top.fill_between(time_axis, pspis, alpha=0.3, color="#EF5350")
    ax_top.plot(time_axis, pspis, color="#EF5350", linewidth=1.5)
    ax_top.set_ylabel("PSPI\n(ground truth)", fontsize=10)
    ax_top.set_title(f"Case Study: Subject {subj_id} — BAUD Pain Score Tracks PSPI Intensity",
                     fontsize=12, fontweight="bold")
    ax_top.set_xlim(0, len(frames))
    ax_top.grid(True, alpha=0.2)
    ax_top.axvspan(0, cal_end, alpha=0.1, color="green")
    ax_top.text(cal_end/2, ax_top.get_ylim()[1]*0.8, "Calibration",
                ha="center", fontsize=8, color="green", fontweight="bold")

    # Bottom: BAUD score
    ax_bot.plot(time_axis, baud_scores, color="#2196F3", linewidth=1.5,
                label="BAUD score")
    ax_bot.plot(time_axis, generic_norm, color="#9E9E9E", linewidth=1,
                alpha=0.5, label="Generic (norm.)")
    ax_bot.set_ylabel("Pain score", fontsize=10)
    ax_bot.set_xlabel("Frame number", fontsize=10)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.axhline(y=0.5, color="black", linestyle=":", alpha=0.2)
    ax_bot.legend(loc="upper right", fontsize=9)
    ax_bot.grid(True, alpha=0.2)
    ax_bot.axvspan(0, cal_end, alpha=0.1, color="green")

    plt.savefig(f"{RESULTS_DIR}/case_study_compact.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: case_study_compact.png")

    # Print summary stats
    from sklearn.metrics import roc_auc_score
    labels = (pspis > 0).astype(int)
    try:
        auc_baud = roc_auc_score(labels, baud_scores)
        auc_gen = roc_auc_score(labels, generic_scores)
    except:
        auc_baud, auc_gen = 0, 0
    print(f"\n  Subject {subj_id} sequence {seq_name}:")
    print(f"    Frames: {len(frames)}, Pain: {sum(labels)}, "
          f"Max PSPI: {max(pspis):.0f}")
    print(f"    BAUD AUC: {auc_baud:.4f}")
    print(f"    Generic AUC: {auc_gen:.4f}")
    print(f"    Personalization advantage: {auc_baud-auc_gen:+.4f}")

    print(f"\n  📤 Share both figures!")


if __name__ == "__main__":
    main()
