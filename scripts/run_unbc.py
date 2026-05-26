#!/usr/bin/env python3
"""
BAUD × UNBC-McMaster Pain Dataset (via Kaggle)

This dataset has REAL shoulder pain patients with PSPI-graded pain levels.
25 subjects, ~48K frames, organized by pain intensity.

Usage on Colab:
    1. Set up Kaggle API key
    2. Run this script
"""
import os, sys, glob, time, json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict, Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = "/content/results"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
BATCH_SIZE = 64
SEED = 42

au_transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================================
# STEP 1: Download and explore dataset
# ============================================================================

def download_dataset():
    """Download from Kaggle."""
    data_dir = "/content/emotionpain"
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"✅ Dataset already at {data_dir}")
        return data_dir

    print("📂 Downloading from Kaggle...")
    os.system("kaggle datasets download -d coder98/emotionpain -p /content/ --unzip")

    # Find the extracted directory
    for candidate in ["/content/emotionpain", "/content/Pain", "/content"]:
        if os.path.exists(candidate):
            # Check for image subdirectories
            subdirs = [d for d in os.listdir(candidate)
                       if os.path.isdir(os.path.join(candidate, d))]
            if subdirs:
                data_dir = candidate
                break

    print(f"✅ Dataset at {data_dir}")
    return data_dir


def explore_dataset(data_dir):
    """Discover dataset structure and print summary."""
    print("\n🔍 Exploring dataset structure...")
    print(f"  Root: {data_dir}")

    class_info = {}
    all_images = []

    for root, dirs, files in os.walk(data_dir):
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        if img_files:
            class_name = os.path.basename(root)
            class_info[class_name] = {
                "path": root,
                "count": len(img_files),
                "files": sorted(img_files),
            }
            # Check image properties
            sample = Image.open(os.path.join(root, img_files[0]))
            class_info[class_name]["size"] = sample.size
            class_info[class_name]["mode"] = sample.mode

            for f in img_files:
                all_images.append((os.path.join(root, f), class_name))

    print(f"\n  Classes found: {len(class_info)}")
    total = 0
    for cls, info in sorted(class_info.items()):
        print(f"    {cls:<20s}: {info['count']:>6d} images, "
              f"size={info['size']}, mode={info['mode']}")
        total += info["count"]
    print(f"    {'TOTAL':<20s}: {total:>6d} images")

    # Try to identify subjects from filenames
    subjects = set()
    for path, cls in all_images[:1000]:
        fname = os.path.basename(path)
        # UNBC filenames typically contain subject IDs like "042-ll042t1afaff012.png"
        parts = fname.split("-")
        if len(parts) >= 1:
            subjects.add(parts[0][:3])  # First 3 chars as subject ID

    if len(subjects) > 1:
        print(f"\n  Unique subject prefixes: {len(subjects)}")
        print(f"  Samples: {sorted(subjects)[:15]}")

    return class_info, all_images


# ============================================================================
# STEP 2: Organize data by subject for BAUD
# ============================================================================

def organize_by_subject(all_images, class_info):
    """
    Group images by subject ID for personalized evaluation.
    UNBC filenames typically: "XXX-llXXXt1afaff012.png" where XXX is subject.
    """
    print("\n📋 Organizing data by subject...")

    # Determine pain level mapping from class names
    class_names = sorted(class_info.keys())
    print(f"  Class names: {class_names}")

    # Auto-detect pain level mapping
    pain_map = {}
    for cls in class_names:
        cls_lower = cls.lower().replace("_", " ").replace("-", " ")
        if "no" in cls_lower and "pain" in cls_lower:
            pain_map[cls] = 0
        elif "no pain" in cls_lower or "nopain" in cls_lower:
            pain_map[cls] = 0
        elif "low" in cls_lower or "mild" in cls_lower or "weak" in cls_lower:
            pain_map[cls] = 1
        elif "moderate" in cls_lower or "medium" in cls_lower or "mid" in cls_lower:
            pain_map[cls] = 2
        elif "severe" in cls_lower or "high" in cls_lower or "strong" in cls_lower:
            pain_map[cls] = 3
        elif "pain" in cls_lower:
            pain_map[cls] = 1  # Generic pain
        else:
            # Try numeric
            try:
                pain_map[cls] = int(cls)
            except:
                pain_map[cls] = -1  # Unknown

    print(f"  Pain level mapping: {pain_map}")

    # Group by subject
    subjects = defaultdict(lambda: defaultdict(list))

    for img_path, class_name in all_images:
        fname = os.path.basename(img_path)
        pain_level = pain_map.get(class_name, -1)
        if pain_level < 0:
            continue

        # Extract subject ID from filename
        # Try different UNBC naming conventions
        subj_id = None

        # Pattern 1: "042-ll042t1afaff012.png"
        parts = fname.split("-")
        if len(parts) >= 1 and parts[0].isdigit():
            subj_id = parts[0]

        # Pattern 2: "subject_042_frame_001.png"
        if subj_id is None:
            for part in fname.replace("_", "-").split("-"):
                if part.isdigit() and len(part) >= 2:
                    subj_id = part
                    break

        # Fallback: use first 3 chars
        if subj_id is None:
            subj_id = fname[:3]

        subjects[subj_id][pain_level].append(img_path)

    # Filter: keep only subjects with both no-pain AND some pain frames
    valid_subjects = {}
    for subj_id, levels in subjects.items():
        if 0 in levels and any(l > 0 for l in levels):
            valid_subjects[subj_id] = dict(levels)

    print(f"\n  Total subjects found: {len(subjects)}")
    print(f"  Valid subjects (have both pain & no-pain): {len(valid_subjects)}")

    for subj_id in sorted(valid_subjects.keys())[:10]:
        levels = valid_subjects[subj_id]
        summary = {f"L{k}": len(v) for k, v in sorted(levels.items())}
        print(f"    Subject {subj_id}: {summary}")
    if len(valid_subjects) > 10:
        print(f"    ... and {len(valid_subjects) - 10} more")

    return valid_subjects, pain_map


# ============================================================================
# STEP 3: Load OpenGraphAU
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
    print(f"✅ OpenGraphAU loaded on {DEVICE}")
    return model


def extract_au_batch(model, image_paths):
    """Extract AUs from a batch of image paths."""
    tensors = []
    valid_idx = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(au_transform(img))
            valid_idx.append(i)
        except:
            pass

    if not tensors:
        return np.array([])

    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        out = model(batch)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return torch.sigmoid(pred).cpu().numpy()


# ============================================================================
# STEP 4: Extract AUs for all subjects
# ============================================================================

def extract_all_aus(model, valid_subjects, max_frames_per_level=200):
    """Extract AUs for all subjects, sampling frames if too many."""
    print(f"\n🔍 Extracting AUs for {len(valid_subjects)} subjects...")
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    au_data = {}
    total_frames = 0

    for idx, (subj_id, levels) in enumerate(sorted(valid_subjects.items())):
        au_data[subj_id] = {}

        for level, paths in sorted(levels.items()):
            # Sample if too many frames
            if len(paths) > max_frames_per_level:
                selected = list(rng.choice(paths, max_frames_per_level, replace=False))
            else:
                selected = paths

            # Extract in batches
            all_aus = []
            for i in range(0, len(selected), BATCH_SIZE):
                batch_paths = selected[i:i + BATCH_SIZE]
                aus = extract_au_batch(model, batch_paths)
                if len(aus) > 0:
                    all_aus.append(aus)

            if all_aus:
                au_data[subj_id][level] = np.concatenate(all_aus)
                total_frames += len(au_data[subj_id][level])

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  Subject {idx+1}/{len(valid_subjects)} "
                  f"({total_frames} frames, {time.time()-t0:.1f}s)")

    print(f"  ✅ Extracted AUs from {total_frames} frames in {time.time()-t0:.1f}s")
    return au_data


# ============================================================================
# STEP 5: Run BAUD experiments
# ============================================================================

def run_baud_experiment(au_data):
    """Run BAUD + baselines on UNBC data with per-subject evaluation."""
    print(f"\n📊 Running BAUD experiment...")

    subjects = sorted(au_data.keys())
    methods = defaultdict(lambda: {"scores": [], "truths": [], "subjects": []})

    for subj_id in subjects:
        levels = au_data[subj_id]

        if 0 not in levels:
            continue
        neutral_aus = levels[0]  # No-pain frames = baseline

        # Calibrate BAUD
        mean_b = np.mean(neutral_aus, axis=0)
        std_b = np.maximum(np.std(neutral_aus, axis=0), 1e-4)

        # Weights
        w = np.ones(neutral_aus.shape[1])
        for idx in PAIN_IDX:
            if idx < len(w): w[idx] = 3.0
        w /= w.sum()

        # Mahalanobis setup
        feat_b = neutral_aus[:, PAIN_IDX]
        mah_mean = np.mean(feat_b, 0)
        mah_cov = np.cov(feat_b, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
        mah_inv = np.linalg.inv(mah_cov)

        for level, pain_aus in sorted(levels.items()):
            is_pain = 1 if level > 0 else 0

            # BAUD
            z = np.maximum((pain_aus - mean_b) / std_b, 0)
            raw = np.array([np.dot(w, f) for f in z])
            baud_s = float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))

            # Generic
            gen_s = float(np.mean([np.mean(f[PAIN_IDX]) for f in pain_aus]))

            # PSPI
            pspi_vals = []
            for f in pain_aus:
                p = f[2] + max(f[4], f[5]) + max(f[6], f[7])
                p += f[17] if len(f) > 17 else 0
                pspi_vals.append(min(p / 2.0, 1.0))
            pspi_s = float(np.mean(pspi_vals))

            # Mahalanobis
            mah_scores = []
            for f in pain_aus[:, PAIN_IDX]:
                diff = f - mah_mean
                d = float(np.sqrt(diff @ mah_inv @ diff))
                mah_scores.append(1.0 / (1.0 + np.exp(-d + 3.0)))
            mah_s = float(np.mean(mah_scores))

            for name, score in [("BAUD (Ours)", baud_s), ("Generic", gen_s),
                                ("PSPI", pspi_s), ("Mahalanobis", mah_s)]:
                methods[name]["scores"].append(score)
                methods[name]["truths"].append(is_pain)
                methods[name]["subjects"].append(subj_id)

    return methods


def compute_and_print_metrics(methods):
    """Compute and print metrics for all methods."""
    print("\n" + "=" * 70)
    print("  UNBC-McMaster RESULTS: Binary Pain Detection (Pain vs No-Pain)")
    print("=" * 70)
    print(f"  {'Method':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} {'N':>6}")
    print("-" * 70)

    all_metrics = {}
    for name, data in methods.items():
        scores = data["scores"]
        truths = data["truths"]

        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t

        preds = [1 if s > best_t else 0 for s in scores]
        try: auc = roc_auc_score(truths, scores)
        except: auc = 0

        acc = accuracy_score(truths, preds)
        all_metrics[name] = {"acc": acc, "f1": best_f1, "auc": auc, "n": len(truths)}
        print(f"  {name:<20} {acc:>8.4f} {best_f1:>8.4f} {auc:>8.4f} {len(truths):>6}")

    print("=" * 70)
    return all_metrics


# ============================================================================
# STEP 6: Visualizations
# ============================================================================

def plot_results(methods, all_metrics, save_dir):
    """Generate plots."""
    # Score distributions
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (name, data) in zip(axes, methods.items()):
        pain_s = [s for s, t in zip(data["scores"], data["truths"]) if t == 1]
        nopain_s = [s for s, t in zip(data["scores"], data["truths"]) if t == 0]
        ax.hist(nopain_s, bins=20, alpha=0.7, label="No Pain", color="#90CAF9", edgecolor="white")
        ax.hist(pain_s, bins=20, alpha=0.7, label="Pain", color="#EF5350", edgecolor="white")
        auc = all_metrics[name]["auc"]
        ax.set_title(f"{name}\nAUC={auc:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("UNBC-McMaster: Pain Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "unbc_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: unbc_distributions.png")

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(all_metrics.keys())
    f1s = [all_metrics[n]["f1"] for n in names]
    aucs = [all_metrics[n]["auc"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.175, f1s, 0.35, label="F1", color="#2196F3", alpha=0.8)
    ax.bar(x + 0.175, aucs, 0.35, label="AUC", color="#FF9800", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score"); ax.set_ylim(0.3, 1.08)
    ax.set_title("UNBC-McMaster: Method Comparison", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for i, (f, a) in enumerate(zip(f1s, aucs)):
        ax.text(i-0.175, f+0.02, f"{f:.3f}", ha="center", fontsize=8)
        ax.text(i+0.175, a+0.02, f"{a:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "unbc_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: unbc_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  BAUD × UNBC-McMaster Shoulder Pain Dataset")
    print("  Real patients, real pain, PSPI-graded intensities")
    print("=" * 65)

    # Download
    data_dir = download_dataset()

    # Explore
    class_info, all_images = explore_dataset(data_dir)

    # Organize by subject
    valid_subjects, pain_map = organize_by_subject(all_images, class_info)

    if not valid_subjects:
        print("\n⚠️  Could not organize by subject.")
        print("  Falling back to class-level evaluation...")
        # Fallback: treat each class as a group
        valid_subjects = {"all": {}}
        for cls, level in pain_map.items():
            if level >= 0 and cls in class_info:
                paths = [os.path.join(class_info[cls]["path"], f)
                         for f in class_info[cls]["files"]]
                valid_subjects["all"][level] = paths

    # Load model
    model = load_model()

    # Extract AUs
    au_data = extract_all_aus(model, valid_subjects)

    # Cache AUs
    cache = {f"{s}_{l}": v for s, levels in au_data.items() for l, v in levels.items()}
    np.savez(os.path.join(RESULTS_DIR, "unbc_extracted_aus.npz"), **cache)
    print(f"  💾 Cached to {RESULTS_DIR}/unbc_extracted_aus.npz")

    # Run experiments
    methods = run_baud_experiment(au_data)
    all_metrics = compute_and_print_metrics(methods)

    # Plots
    print("\n📈 Generating plots...")
    plot_results(methods, all_metrics, RESULTS_DIR)

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "unbc_metrics.txt"), "w") as f:
        f.write("BAUD × UNBC-McMaster Results\n")
        f.write("=" * 70 + "\n")
        for name, m in all_metrics.items():
            f.write(f"{name:<20} Acc={m['acc']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}\n")
    print("  Saved: unbc_metrics.txt")

    print(f"\n{'=' * 65}")
    print(f"  ✅ UNBC EXPERIMENT COMPLETE")
    print(f"{'=' * 65}")
    print(f"  📤 Share console output + plots!")


if __name__ == "__main__":
    main()
