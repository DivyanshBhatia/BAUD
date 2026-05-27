#!/usr/bin/env python3
"""
Two-Subject Case Study: Success + Struggle
Shows BAUD working well AND where it struggles, with explanations.
Addresses cherry-picking concern.

Run on Colab:
    python scripts/run_case_study_v2.py
"""
import os, sys, numpy as np, torch, time
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
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

def get_all_sequences():
    """Get all subject sequences."""
    subjects = {}
    for subj_folder in sorted(os.listdir(IMAGES_DIR)):
        subj_id = subj_folder.split("-")[0]
        subj_img = os.path.join(IMAGES_DIR, subj_folder)
        subj_lbl = os.path.join(LABELS_DIR, subj_folder)
        if not os.path.isdir(subj_img) or not os.path.isdir(subj_lbl): continue
        for seq in sorted(os.listdir(subj_img)):
            seq_img = os.path.join(subj_img, seq)
            seq_lbl = os.path.join(subj_lbl, seq)
            if not os.path.isdir(seq_img) or not os.path.isdir(seq_lbl): continue
            frames = []
            for img_file in sorted(os.listdir(seq_img)):
                if not img_file.endswith(".png"): continue
                lbl_file = os.path.join(seq_lbl, img_file.replace(".png","")+"_facs.txt")
                if os.path.exists(lbl_file):
                    try:
                        with open(lbl_file) as f: pspi = float(f.read().strip())
                        frames.append({"path": os.path.join(seq_img, img_file), "pspi": pspi})
                    except: pass
            if len(frames) >= 50:
                key = f"{subj_id}_{seq}"
                subjects[key] = {"subj": subj_id, "seq": seq, "frames": frames}
    return subjects

def score_sequence(model, frames):
    """Extract AUs and compute BAUD + Generic scores for a sequence."""
    aus = extract_aus(model, [f["path"] for f in frames])
    pspis = np.array([f["pspi"] for f in frames])
    labels = (pspis > 0).astype(int)

    if sum(labels) < 3 or sum(1-labels) < 3:
        return None

    # Baseline from neutral frames
    neutral = aus[labels == 0]
    mean_b = np.mean(neutral[:20] if len(neutral) >= 20 else neutral, 0)
    std_b = np.maximum(np.std(neutral[:20] if len(neutral) >= 20 else neutral, 0), 1e-4)
    
    w = np.ones(aus.shape[1])
    for i in PAIN_IDX: w[i] = 3.0
    w /= w.sum()

    baud_scores = []
    for f in aus:
        z = np.maximum((f - mean_b) / std_b, 0)
        baud_scores.append(1.0 / (1.0 + np.exp(-np.dot(w, z) + 2.0)))
    baud_scores = np.array(baud_scores)

    generic_scores = np.array([np.mean(f[PAIN_IDX]) for f in aus])
    gen_norm = (generic_scores - generic_scores.min()) / (generic_scores.max() - generic_scores.min() + 1e-8)

    try:
        auc_baud = roc_auc_score(labels, baud_scores)
        auc_gen = roc_auc_score(labels, generic_scores)
    except:
        return None

    # Pain contamination in first 50 frames
    pain_in_50 = sum(1 for f in frames[:50] if f["pspi"] > 0)

    return {
        "pspis": pspis, "baud": baud_scores, "generic": gen_norm,
        "labels": labels, "auc_baud": auc_baud, "auc_gen": auc_gen,
        "pain_in_50": pain_in_50, "max_pspi": max(pspis),
        "n_frames": len(frames), "n_pain": sum(labels),
    }


def main():
    print("=" * 65)
    print("  TWO-SUBJECT CASE STUDY: Success + Struggle")
    print("=" * 65)

    model = load_model()
    print("  ✅ Model loaded")

    sequences = get_all_sequences()
    print(f"  {len(sequences)} sequences found")

    # Score all sequences
    print("  Scoring all sequences...")
    scored = {}
    t0 = time.time()
    for i, (key, data) in enumerate(sorted(sequences.items())):
        result = score_sequence(model, data["frames"])
        if result:
            scored[key] = {**result, "subj": data["subj"], "seq": data["seq"]}
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(sequences)} ({time.time()-t0:.1f}s)")

    print(f"  ✅ Scored {len(scored)} valid sequences")

    # Find best success and best struggle
    by_advantage = sorted(scored.items(), key=lambda x: x[1]["auc_baud"] - x[1]["auc_gen"])

    # Success: high BAUD AUC AND large advantage over Generic
    successes = [(k, v) for k, v in scored.items()
                 if v["auc_baud"] > 0.75 and v["auc_baud"] - v["auc_gen"] > 0.1
                 and v["max_pspi"] >= 4 and v["n_pain"] >= 10]
    if successes:
        success_key, success = max(successes, key=lambda x: x[1]["auc_baud"] - x[1]["auc_gen"])
    else:
        success_key, success = by_advantage[-1]

    # Struggle: low BAUD AUC or Generic beats BAUD
    struggles = [(k, v) for k, v in scored.items()
                 if v["auc_baud"] < 0.6 and v["n_pain"] >= 5]
    if struggles:
        struggle_key, struggle = min(struggles, key=lambda x: x[1]["auc_baud"])
    else:
        struggle_key, struggle = by_advantage[0]

    print(f"\n  SUCCESS: {success_key}")
    print(f"    BAUD={success['auc_baud']:.3f}, Generic={success['auc_gen']:.3f}, "
          f"gap={success['auc_baud']-success['auc_gen']:+.3f}")
    print(f"    {success['n_frames']} frames, {success['n_pain']} pain, "
          f"max PSPI={success['max_pspi']:.0f}, pain in first 50={success['pain_in_50']}")

    print(f"\n  STRUGGLE: {struggle_key}")
    print(f"    BAUD={struggle['auc_baud']:.3f}, Generic={struggle['auc_gen']:.3f}, "
          f"gap={struggle['auc_baud']-struggle['auc_gen']:+.3f}")
    print(f"    {struggle['n_frames']} frames, {struggle['n_pain']} pain, "
          f"max PSPI={struggle['max_pspi']:.0f}, pain in first 50={struggle['pain_in_50']}")

    # ── Generate Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 6),
                              gridspec_kw={"hspace": 0.4, "wspace": 0.05},
                              sharex="col")

    for col, (key, data, title_prefix) in enumerate([
        (success_key, success, "Success"),
        (struggle_key, struggle, "Struggle"),
    ]):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        t = np.arange(data["n_frames"])
        pain_mask = data["pspis"] > 0

        # Top: PSPI
        ax_top.fill_between(t, data["pspis"], alpha=0.3, color="#EF5350")
        ax_top.plot(t, data["pspis"], color="#EF5350", linewidth=1.2)
        ax_top.set_ylabel("PSPI" if col == 0 else "", fontsize=10)
        gap = data["auc_baud"] - data["auc_gen"]
        ax_top.set_title(
            f"{title_prefix}: Subject {data['subj']}\n"
            f"BAUD AUC={data['auc_baud']:.3f}, Generic={data['auc_gen']:.3f} "
            f"({gap:+.3f})",
            fontsize=11, fontweight="bold",
            color="#2196F3" if title_prefix == "Success" else "#EF5350")
        ax_top.set_xlim(0, data["n_frames"])
        ax_top.grid(True, alpha=0.15)

        # Calibration shading
        neutral_idx = np.where(data["labels"] == 0)[0]
        cal_end = neutral_idx[19] if len(neutral_idx) >= 20 else 20
        ax_top.axvspan(0, cal_end, alpha=0.1, color="green")
        if col == 0:
            ax_top.text(cal_end/2, ax_top.get_ylim()[1]*0.85, "Cal.",
                        ha="center", fontsize=7, color="green", fontweight="bold")

        # Bottom: BAUD vs Generic
        ax_bot.plot(t, data["baud"], color="#2196F3", linewidth=1.2,
                    label="BAUD" if col == 0 else "")
        ax_bot.plot(t, data["generic"], color="#9E9E9E", linewidth=0.8,
                    alpha=0.5, label="Generic" if col == 0 else "")
        ax_bot.axhline(y=0.5, color="black", linestyle=":", alpha=0.2)
        ax_bot.set_ylabel("Pain score" if col == 0 else "", fontsize=10)
        ax_bot.set_xlabel("Frame number", fontsize=10)
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_xlim(0, data["n_frames"])
        ax_bot.grid(True, alpha=0.15)
        ax_bot.axvspan(0, cal_end, alpha=0.1, color="green")
        ax_bot.fill_between(t, 0, 1, where=pain_mask, alpha=0.06, color="red")

        # Annotate why struggle fails
        if title_prefix == "Struggle":
            if data["pain_in_50"] > 5:
                ax_top.text(0.5, 0.5, f"⚠ {data['pain_in_50']} pain frames\nin calibration window",
                           transform=ax_top.transAxes, fontsize=8, color="red",
                           ha="center", va="center",
                           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
            else:
                ax_bot.text(0.5, 0.85, "Low-expressiveness subject:\nsubtle pain signal",
                           transform=ax_bot.transAxes, fontsize=8, color="red",
                           ha="center", va="top",
                           bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    axes[1, 0].legend(loc="upper left", fontsize=9)
    plt.savefig(f"{RESULTS_DIR}/case_study_dual.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  ✅ Saved: case_study_dual.png")

    # Also print all subjects ranked by personalization advantage
    print(f"\n  All sequences by personalization advantage:")
    print(f"  {'Key':<25} {'BAUD':>7} {'Generic':>8} {'Gap':>7} {'Pain/50':>8}")
    print(f"  {'-'*58}")
    for key, data in sorted(scored.items(), key=lambda x: -(x[1]["auc_baud"]-x[1]["auc_gen"])):
        gap = data["auc_baud"] - data["auc_gen"]
        print(f"  {key:<25} {data['auc_baud']:>7.3f} {data['auc_gen']:>8.3f} "
              f"{gap:>+7.3f} {data['pain_in_50']:>8}")


if __name__ == "__main__":
    main()
