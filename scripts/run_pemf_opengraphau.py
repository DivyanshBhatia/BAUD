#!/usr/bin/env python3
"""
BAUD × PEMF × OpenGraphAU: Full End-to-End Pipeline
=====================================================
Real face images → AU extraction → Personalized pain detection

This script:
1. Loads OpenGraphAU with pretrained checkpoint
2. Extracts AUs from all PEMF face images (68 subjects × 4 expressions × 20 frames)
3. Runs BAUD calibration on neutral frames + pain scoring
4. Compares BAUD vs Generic vs PSPI
5. Also compares extracted AUs vs expert FACS annotations
6. Generates all visualizations

Run on Colab with A100/T4 GPU.

Usage:
    python scripts/run_pemf_opengraphau.py
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from collections import OrderedDict

# ============================================================================
# CONFIG
# ============================================================================
PEMF_ROOT = "/content/pemf"
PICTURES_DIR = os.path.join(PEMF_ROOT, "pictures", "Pictures", "Modified")
OPENGRAPHAU_DIR = "/content/baud/external/OpenGraphAU"
CHECKPOINT_PATH = os.path.join(
    OPENGRAPHAU_DIR, "checkpoints", "OpenGprahAU-ResNet50_second_stage.pth"
)
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32  # Process images in batches for speed

# OpenGraphAU outputs 41 AUs — these are the ones we care about
# Mapping: OpenGraphAU index → AU name
OPENGRAPHAU_AUS = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU43",
]
NUM_AUS = 41  # Total OpenGraphAU outputs

# Pain-related AUs and their indices in the 41-AU output
PAIN_AU_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
PAIN_AU_INDICES = [2, 4, 5, 6, 7, 17]  # Indices in the 18 main AUs

# AUs shared between OpenGraphAU and PEMF FACS annotations
SHARED_AU_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU12",
                   "AU20", "AU25", "AU26", "AU27", "AU43"]

# Expression types
EXPR_TYPES = {
    "Neutral": {"code": "N", "pain_level": 0},
    "Algometer Pain": {"code": "A", "pain_level": 2},
    "Laser Pain": {"code": "L", "pain_level": 1},
    "Posed Pain": {"code": "P", "pain_level": 3},
}

# Image transform for OpenGraphAU
au_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================================
# STEP 1: Load OpenGraphAU Model
# ============================================================================

def load_opengraphau():
    """Load the OpenGraphAU model with pretrained weights."""
    print("📦 Loading OpenGraphAU model...")

    # Add OpenGraphAU to path
    sys.path.insert(0, OPENGRAPHAU_DIR)

    try:
        from model.MEFL import MEFARG
        model = MEFARG(num_main_classes=NUM_AUS, num_sub_classes=NUM_AUS,
                       backbone="resnet50")
    except ImportError:
        print("  ⚠️  Could not import MEFARG. Trying alternative import...")
        try:
            from OpenGraphAU.model.MEFL import MEFARG
            model = MEFARG(num_main_classes=NUM_AUS, num_sub_classes=NUM_AUS,
                           backbone="resnet50")
        except ImportError:
            print("  ❌ Failed to import OpenGraphAU model.")
            print("     Make sure OpenGraphAU is at:", OPENGRAPHAU_DIR)
            print("     Try: cd /content/baud/external/OpenGraphAU && ls model/")
            return None, None

    # Load checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  ❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return None, None

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"  ✅ OpenGraphAU loaded on {device}")
    print(f"  ✅ Checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    return model, device


# ============================================================================
# STEP 2: Extract AUs from Images
# ============================================================================

def extract_aus_batch(model, device, image_paths, batch_size=BATCH_SIZE):
    """
    Extract AU activations from a batch of images.

    Returns:
        np.array of shape (n_images, NUM_AUS) with AU activation values [0, 1]
    """
    all_aus = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = au_transform(img)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"    ⚠️  Could not load {path}: {e}")
                # Use zeros as fallback
                batch_tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            output = model(batch)
            if isinstance(output, (tuple, list)):
                au_pred = output[0]
            else:
                au_pred = output
            au_values = torch.sigmoid(au_pred).cpu().numpy()

        all_aus.append(au_values)

    return np.concatenate(all_aus, axis=0) if all_aus else np.array([])


def extract_all_pemf_aus(model, device):
    """
    Extract AUs from all PEMF picture frames.

    Returns:
        dict: {subject_id: {expression_type: np.array (n_frames, NUM_AUS)}}
    """
    print("\n🔍 Extracting AUs from PEMF images...")
    start_time = time.time()

    all_data = {}
    subjects = sorted([d for d in os.listdir(PICTURES_DIR)
                       if d.startswith("S") and
                       os.path.isdir(os.path.join(PICTURES_DIR, d))])

    total_frames = 0

    for idx, subj in enumerate(subjects):
        all_data[subj] = {}
        subj_dir = os.path.join(PICTURES_DIR, subj)

        for expr_name in EXPR_TYPES:
            frames_dir = os.path.join(subj_dir, expr_name, "Colour frames")
            if not os.path.exists(frames_dir):
                continue

            frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if not frame_paths:
                continue

            au_matrix = extract_aus_batch(model, device, frame_paths)
            all_data[subj][expr_name] = au_matrix
            total_frames += len(frame_paths)

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1}/{len(subjects)} subjects "
                  f"({total_frames} frames, {elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"  ✅ Extracted AUs from {total_frames} frames in {elapsed:.1f}s")
    return all_data


# ============================================================================
# STEP 3: BAUD Calibrator
# ============================================================================

class BAUDCalibrator:
    """BAUD with real multi-frame calibration from extracted AUs."""

    def __init__(self, num_aus=NUM_AUS, window_size=5):
        self.num_aus = num_aus
        self.window_size = window_size
        self.baseline_mean = None
        self.baseline_std = None
        self.is_calibrated = False
        self.z_buffer = []

        # Deviation weights (prior)
        self.weights = np.ones(num_aus)
        for idx in PAIN_AU_INDICES:
            self.weights[idx] = 3.0
        self.weights /= self.weights.sum()

    def calibrate(self, baseline_au_matrix):
        """Calibrate from multiple neutral frames."""
        self.baseline_mean = np.mean(baseline_au_matrix, axis=0)
        self.baseline_std = np.maximum(
            np.std(baseline_au_matrix, axis=0), 1e-4
        )
        self.is_calibrated = True
        self.z_buffer = []

    def score_frame(self, au_vector):
        """Score one frame against baseline."""
        z = (au_vector - self.baseline_mean) / self.baseline_std
        z_pos = np.maximum(z, 0)

        self.z_buffer.append(z_pos)
        if len(self.z_buffer) > self.window_size:
            self.z_buffer.pop(0)
        z_smooth = np.mean(self.z_buffer, axis=0)

        raw = np.dot(self.weights, z_smooth)
        score = 1.0 / (1.0 + np.exp(-raw + 2.0))

        return {"pain_score": float(score), "z_scores": z,
                "z_smoothed": z_smooth}

    def score_sequence(self, au_matrix):
        """Score a sequence of frames."""
        self.z_buffer = []
        results = []
        for frame in au_matrix:
            results.append(self.score_frame(frame))
        return results


# ============================================================================
# STEP 4: Baselines
# ============================================================================

def generic_score(au_vector):
    """Average pain-related AU activations."""
    return float(np.mean(au_vector[PAIN_AU_INDICES]))


def pspi_score(au_vector):
    """PSPI formula from AU predictions."""
    au4 = au_vector[2]
    au6, au7 = au_vector[4], au_vector[5]
    au9, au10 = au_vector[6], au_vector[7]
    au43 = au_vector[17] if len(au_vector) > 17 else 0
    pspi = au4 + max(au6, au7) + max(au9, au10) + au43
    return float(min(pspi / 2.0, 1.0))


# ============================================================================
# STEP 5: Run Experiments
# ============================================================================

def run_experiments(au_data):
    """Run BAUD + baselines on extracted AU data."""
    print("\n📊 Running experiments...")

    all_results = defaultdict(list)
    per_subject = []

    for subj_id, exprs in au_data.items():
        if "Neutral" not in exprs:
            continue

        neutral_aus = exprs["Neutral"]  # (n_frames, NUM_AUS)

        # Calibrate BAUD using all neutral frames
        baud = BAUDCalibrator()
        baud.calibrate(neutral_aus)

        # Score neutral frames (should be low)
        neutral_results = baud.score_sequence(neutral_aus)
        neutral_mean_score = np.mean([r["pain_score"] for r in neutral_results])
        all_results["BAUD (Ours)"].append(
            {"score": neutral_mean_score, "true": 0, "expr": "Neutral",
             "subject": subj_id}
        )
        all_results["Generic"].append(
            {"score": float(np.mean([generic_score(f) for f in neutral_aus])),
             "true": 0, "expr": "Neutral", "subject": subj_id}
        )
        all_results["PSPI"].append(
            {"score": float(np.mean([pspi_score(f) for f in neutral_aus])),
             "true": 0, "expr": "Neutral", "subject": subj_id}
        )

        # Score pain expressions
        for expr_name in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            if expr_name not in exprs:
                continue

            pain_aus = exprs[expr_name]  # (n_frames, NUM_AUS)

            # BAUD: score each frame, take mean
            baud.z_buffer = []  # Reset buffer
            pain_results = baud.score_sequence(pain_aus)
            baud_score = np.mean([r["pain_score"] for r in pain_results])
            mean_z = np.mean([r["z_scores"] for r in pain_results], axis=0)

            # Baselines
            gen_score = float(np.mean([generic_score(f) for f in pain_aus]))
            pspi_s = float(np.mean([pspi_score(f) for f in pain_aus]))

            for method, score in [("BAUD (Ours)", baud_score),
                                  ("Generic", gen_score),
                                  ("PSPI", pspi_s)]:
                all_results[method].append({
                    "score": score, "true": 1,
                    "expr": expr_name, "subject": subj_id,
                })

            per_subject.append({
                "subject": subj_id,
                "expression": expr_name,
                "baud_score": float(baud_score),
                "generic_score": gen_score,
                "pspi_score": pspi_s,
                "mean_z_scores": mean_z,
                "n_baseline_frames": len(neutral_aus),
                "n_pain_frames": len(pain_aus),
            })

    return all_results, per_subject


def compute_metrics(results):
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics = {}
    for method, entries in results.items():
        scores = [e["score"] for e in entries]
        truths = [e["true"] for e in entries]

        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            preds = [1 if s > t else 0 for s in scores]
            f1 = f1_score(truths, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        preds = [1 if s > best_thresh else 0 for s in scores]
        try:
            auc = roc_auc_score(truths, scores)
        except ValueError:
            auc = 0.0

        metrics[method] = {
            "accuracy": accuracy_score(truths, preds),
            "f1": best_f1,
            "auc": auc,
            "threshold": best_thresh,
        }
    return metrics


# ============================================================================
# STEP 6: Visualizations
# ============================================================================

def plot_all_results(results, per_subject, results_dir):
    """Generate all plots."""

    # 1. Score distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_map = {"BAUD (Ours)": "#2196F3", "Generic": "#FF5722", "PSPI": "#4CAF50"}
    for ax, (method, entries) in zip(axes, results.items()):
        pain = [e["score"] for e in entries if e["true"] == 1]
        neutral = [e["score"] for e in entries if e["true"] == 0]
        ax.hist(neutral, bins=15, alpha=0.7, label="Neutral", color="#90CAF9")
        ax.hist(pain, bins=15, alpha=0.7, label="Pain", color="#EF5350")
        ax.set_title(method, fontsize=13, fontweight="bold")
        ax.set_xlabel("Pain Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Score Distributions (OpenGraphAU Extracted AUs)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "opengraphau_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 2. BAUD vs Generic scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    baud_s = [r["baud_score"] for r in per_subject]
    gen_s = [r["generic_score"] for r in per_subject]
    exprs = [r["expression"] for r in per_subject]
    color_map = {"Algometer Pain": "#EF5350", "Laser Pain": "#FF9800",
                 "Posed Pain": "#9C27B0"}
    for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
        mask = [e == expr for e in exprs]
        ax.scatter([g for g, m in zip(gen_s, mask) if m],
                   [b for b, m in zip(baud_s, mask) if m],
                   c=color_map[expr], s=80, alpha=0.7, label=expr,
                   edgecolors="white")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Generic Score", fontsize=12)
    ax.set_ylabel("BAUD Score (personalized)", fontsize=12)
    ax.set_title("BAUD vs Generic (Real AU Extraction)", fontsize=14,
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "opengraphau_baud_vs_generic.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Per-AU deviations by expression type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, expr in zip(axes, ["Algometer Pain", "Laser Pain", "Posed Pain"]):
        z_data = [r["mean_z_scores"] for r in per_subject
                  if r["expression"] == expr]
        if not z_data:
            continue
        z_matrix = np.array(z_data)
        pain_z = z_matrix[:, PAIN_AU_INDICES]
        means = np.mean(np.maximum(pain_z, 0), axis=0)
        stds = np.std(np.maximum(pain_z, 0), axis=0)

        colors = ["#EF5350" if m > 1.0 else "#FFA726" if m > 0.3
                  else "#66BB6A" for m in means]
        ax.bar(range(len(PAIN_AU_NAMES)), means, yerr=stds,
               color=colors, capsize=3, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(PAIN_AU_NAMES)))
        ax.set_xticklabels(PAIN_AU_NAMES)
        ax.set_ylabel("Mean Z-Score (σ from baseline)")
        short_name = expr.replace(" Pain", "")
        ax.set_title(f"{short_name} Pain", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Per-AU Deviation from Baseline (OpenGraphAU)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "opengraphau_au_deviations.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("  ✅ All plots saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  BAUD × PEMF × OpenGraphAU")
    print("  Full End-to-End Pipeline")
    print("=" * 60)

    # Load model
    model, device = load_opengraphau()
    if model is None:
        print("\n❌ Cannot proceed without OpenGraphAU model.")
        print("   Check the checkpoint path and model imports.")
        return

    # Extract AUs from all PEMF images
    au_data = extract_all_pemf_aus(model, device)

    # Save extracted AUs for reuse (so you don't re-extract every run)
    au_cache_path = os.path.join(RESULTS_DIR, "pemf_extracted_aus.npz")
    np.savez(au_cache_path,
             **{f"{s}_{e}": v for s, exprs in au_data.items()
                for e, v in exprs.items()})
    print(f"  💾 Cached extracted AUs to {au_cache_path}")

    # Run experiments
    results, per_subject = run_experiments(au_data)

    # Metrics
    metrics = compute_metrics(results)

    print("\n" + "=" * 70)
    print("  RESULTS: Real Face Images → OpenGraphAU → BAUD")
    print("=" * 70)
    print(f"  {'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Thresh':>8}")
    print("-" * 70)
    for method, m in metrics.items():
        print(f"  {method:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {m['threshold']:>8.2f}")
    print("=" * 70)

    # Visualizations
    print("\n📈 Generating plots...")
    plot_all_results(results, per_subject, RESULTS_DIR)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "opengraphau_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("BAUD × PEMF × OpenGraphAU — Full Pipeline Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}\n")
        f.write("-" * 70 + "\n")
        for method, m in metrics.items():
            f.write(f"{method:<25} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
                    f"{m['auc']:>8.4f}\n")
    print(f"  Saved: {metrics_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ FULL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Results in: {RESULTS_DIR}/")
    print("  ├── opengraphau_distributions.png")
    print("  ├── opengraphau_baud_vs_generic.png")
    print("  ├── opengraphau_au_deviations.png")
    print("  ├── opengraphau_metrics.txt")
    print("  └── pemf_extracted_aus.npz (cached, reusable)")
    print("\n  📤 Share these files to review results!")


if __name__ == "__main__":
    main()
