#!/usr/bin/env python3
"""
BAUD × SynPAIN: Test personalized pain detection on synthetic data.

SynPAIN has 5,355 paired images: each identity has a NEUTRAL face
and an EXPRESSIVE face (either pain or non-pain).

This tests a HARDER task than PEMF: can BAUD distinguish pain expressions
from non-pain expressions (surprise, disgust, etc.)? Both deviate from
baseline, but only pain should activate the specific pain AU pattern.

Usage on Colab:
    pip install datasets huggingface_hub
    python scripts/run_synpain.py
"""
import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = "/content/results"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_AUS = 41
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
BATCH_SIZE = 64
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

au_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================================
# STEP 1: Load SynPAIN from HuggingFace
# ============================================================================

def load_synpain():
    """Download and load SynPAIN dataset."""
    print("📂 Loading SynPAIN from HuggingFace...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("TaatiTeam/SynPAIN", split="train")
        print(f"  Loaded {len(dataset)} samples")
        print(f"  Features: {list(dataset.features.keys())}")
        # Print a sample to understand structure
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        for k, v in sample.items():
            if isinstance(v, str):
                print(f"    {k}: {v}")
            elif isinstance(v, (int, float)):
                print(f"    {k}: {v}")
            elif hasattr(v, 'size'):
                print(f"    {k}: Image {v.size}")
        return dataset
    except Exception as e:
        print(f"  ⚠️  Could not load from HuggingFace: {e}")
        print("  Trying alternative download...")
        return None


def parse_synpain(dataset):
    """Parse SynPAIN into neutral/expressive pairs grouped by identity."""
    print("\n🔍 Parsing SynPAIN pairs...")

    pairs = []
    feature_names = list(dataset.features.keys())
    print(f"  Available features: {feature_names}")

    for i, sample in enumerate(dataset):
        pair = {"index": i}

        # Extract images and metadata
        # SynPAIN structure: each sample has neutral image, expressive image,
        # and metadata (expression type, ethnicity, age, gender)
        for key in feature_names:
            pair[key] = sample[key]

        pairs.append(pair)

        if i == 0:
            print(f"  First sample structure:")
            for k, v in pair.items():
                if isinstance(v, Image.Image):
                    print(f"    {k}: PIL Image {v.size}")
                elif isinstance(v, str):
                    print(f"    {k}: '{v}'")
                else:
                    print(f"    {k}: {type(v).__name__}")

    print(f"  Parsed {len(pairs)} pairs")
    return pairs


# ============================================================================
# STEP 2: Load OpenGraphAU
# ============================================================================

def load_model():
    """Load OpenGraphAU encoder."""
    print("\n📦 Loading OpenGraphAU...")
    sys.path.insert(0, OPENGRAPHAU_DIR)

    try:
        from model.MEFL import MEFARG
    except ImportError:
        print("  ❌ Could not import OpenGraphAU")
        print(f"  Check path: {OPENGRAPHAU_DIR}")
        return None, None

    model = MEFARG(num_main_classes=27, num_sub_classes=14, backbone="resnet50")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=False)
    model.eval()
    model = model.to(DEVICE)
    print(f"  ✅ OpenGraphAU loaded on {DEVICE}")
    return model, DEVICE


def extract_au_single(model, device, pil_image):
    """Extract AU vector from a single PIL image."""
    tensor = au_transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return torch.sigmoid(pred).cpu().numpy().flatten()


def extract_au_batch(model, device, pil_images):
    """Extract AU vectors from a batch of PIL images."""
    tensors = [au_transform(img.convert("RGB")) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        out = model(batch)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return torch.sigmoid(pred).cpu().numpy()


# ============================================================================
# STEP 3: BAUD Scoring
# ============================================================================

def baud_score_pair(neutral_au, expressive_au):
    """
    Score an expressive face against a neutral baseline.
    Since we have only 1 neutral frame, we use a fixed small std.
    """
    # With single frame, estimate std from the AU value itself
    # (larger AU values tend to have more variance)
    estimated_std = np.maximum(np.abs(neutral_au) * 0.3 + 0.01, 0.01)

    z = (expressive_au - neutral_au) / estimated_std
    z_pos = np.maximum(z, 0)

    # Pain-weighted scoring
    w = np.ones(len(z_pos))
    for idx in PAIN_IDX:
        if idx < len(w):
            w[idx] = 3.0
    w /= w.sum()

    raw = np.dot(w, z_pos)
    score = 1.0 / (1.0 + np.exp(-raw + 2.0))
    return float(score), z_pos


def generic_score_pair(expressive_au):
    """Non-personalized: raw pain AU activation."""
    return float(np.mean(expressive_au[PAIN_IDX]))


# ============================================================================
# STEP 4: Run Experiments
# ============================================================================

def run_experiment(dataset, model, device):
    """Run BAUD on all SynPAIN pairs."""
    print("\n📊 Running BAUD experiment on SynPAIN...")

    feature_names = list(dataset.features.keys())

    # Find image and label columns
    img_cols = [c for c in feature_names if "image" in c.lower() or "img" in c.lower()
                or "neutral" in c.lower() or "pain" in c.lower() or "expressive" in c.lower()]
    label_cols = [c for c in feature_names if "label" in c.lower() or "type" in c.lower()
                  or "expression" in c.lower() or "class" in c.lower()]

    print(f"  Image columns: {img_cols}")
    print(f"  Label columns: {label_cols}")
    print(f"  All columns: {feature_names}")

    baud_results = []
    generic_results = []
    z_score_data = {"pain": [], "non_pain": []}

    t0 = time.time()
    n_processed = 0
    n_errors = 0

    for i in range(len(dataset)):
        try:
            sample = dataset[i]

            # Try to find neutral and expressive images
            # SynPAIN structure varies — adapt based on actual features
            neutral_img = None
            expr_img = None
            is_pain = None

            # Strategy 1: separate neutral/expressive columns
            for col in feature_names:
                val = sample[col]
                if isinstance(val, Image.Image):
                    col_lower = col.lower()
                    if "neutral" in col_lower:
                        neutral_img = val
                    elif "pain" in col_lower or "expressive" in col_lower or "expr" in col_lower:
                        expr_img = val

            # Strategy 2: single image column with label
            if neutral_img is None and expr_img is None:
                for col in feature_names:
                    if isinstance(sample[col], Image.Image):
                        # If there's only one image, we can't do baseline comparison
                        # Skip this sample
                        pass

            # Find pain/non-pain label
            for col in feature_names:
                val = sample[col]
                if isinstance(val, str):
                    val_lower = val.lower()
                    if "pain" in val_lower and "non" not in val_lower:
                        is_pain = 1
                    elif "non" in val_lower or "neutral" in val_lower or "other" in val_lower:
                        is_pain = 0
                elif isinstance(val, (int, float)) and col.lower() in ["label", "class", "pain"]:
                    is_pain = int(val)

            if neutral_img is None or expr_img is None or is_pain is None:
                if i == 0:
                    print(f"\n  ⚠️  Could not parse sample 0. Printing all fields:")
                    for k in feature_names:
                        v = sample[k]
                        print(f"    {k}: {type(v).__name__} = {v if not isinstance(v, Image.Image) else v.size}")
                    print("\n  Will try adaptive parsing...")

                    # Adaptive: try all image columns as potential pairs
                    all_imgs = [(k, sample[k]) for k in feature_names if isinstance(sample[k], Image.Image)]
                    if len(all_imgs) >= 2:
                        neutral_img = all_imgs[0][1]
                        expr_img = all_imgs[1][1]
                        print(f"  Using {all_imgs[0][0]} as neutral, {all_imgs[1][0]} as expressive")

                        # Try to find label from string fields
                        for k in feature_names:
                            v = sample[k]
                            if isinstance(v, str) and ("pain" in v.lower()):
                                is_pain = 0 if "non" in v.lower() else 1
                                print(f"  Label from '{k}': {v} → is_pain={is_pain}")
                                break

                if neutral_img is None or expr_img is None or is_pain is None:
                    n_errors += 1
                    if n_errors <= 3:
                        pass  # Skip silently
                    continue

            # Extract AUs
            neutral_au = extract_au_single(model, device, neutral_img)
            expr_au = extract_au_single(model, device, expr_img)

            # BAUD score
            baud_s, z_pos = baud_score_pair(neutral_au, expr_au)
            baud_results.append({"score": baud_s, "true": is_pain})

            # Generic score
            gen_s = generic_score_pair(expr_au)
            generic_results.append({"score": gen_s, "true": is_pain})

            # Collect z-scores for analysis
            if is_pain == 1:
                z_score_data["pain"].append(z_pos)
            else:
                z_score_data["non_pain"].append(z_pos)

            n_processed += 1
            if n_processed % 500 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {n_processed} pairs ({elapsed:.1f}s)")

        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  Error on sample {i}: {e}")

    elapsed = time.time() - t0
    print(f"\n  ✅ Processed {n_processed} pairs in {elapsed:.1f}s ({n_errors} errors)")

    return baud_results, generic_results, z_score_data


# ============================================================================
# STEP 5: Metrics and Visualization
# ============================================================================

def compute_metrics(results, method_name):
    scores = [r["score"] for r in results]
    truths = [r["true"] for r in results]

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds = [1 if s > best_t else 0 for s in scores]
    try:
        auc = roc_auc_score(truths, scores)
    except:
        auc = 0.0

    return {
        "method": method_name,
        "acc": accuracy_score(truths, preds),
        "f1": best_f1,
        "auc": auc,
        "thresh": best_t,
        "n_pain": sum(truths),
        "n_nonpain": len(truths) - sum(truths),
    }


def plot_results(baud_results, generic_results, z_score_data, save_dir):
    """Generate visualization plots."""

    # 1. Score distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, name in [(ax1, baud_results, "BAUD"),
                               (ax2, generic_results, "Generic")]:
        pain_scores = [r["score"] for r in results if r["true"] == 1]
        nonpain_scores = [r["score"] for r in results if r["true"] == 0]

        ax.hist(nonpain_scores, bins=20, alpha=0.7, label="Non-Pain",
                color="#90CAF9", edgecolor="white")
        ax.hist(pain_scores, bins=20, alpha=0.7, label="Pain",
                color="#EF5350", edgecolor="white")
        ax.set_title(f"{name}: Pain vs Non-Pain", fontsize=13, fontweight="bold")
        ax.set_xlabel("Pain Score")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("SynPAIN: Score Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "synpain_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: synpain_distributions.png")

    # 2. Per-AU z-score comparison (pain vs non-pain)
    if z_score_data["pain"] and z_score_data["non_pain"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        pain_z = np.array(z_score_data["pain"])
        nonpain_z = np.array(z_score_data["non_pain"])

        pain_mean = np.mean(pain_z[:, PAIN_IDX], axis=0)
        nonpain_mean = np.mean(nonpain_z[:, PAIN_IDX], axis=0)
        pain_std = np.std(pain_z[:, PAIN_IDX], axis=0)
        nonpain_std = np.std(nonpain_z[:, PAIN_IDX], axis=0)

        x = np.arange(len(PAIN_NAMES))
        width = 0.35
        ax.bar(x - width/2, pain_mean, width, yerr=pain_std, capsize=3,
               label="Pain", color="#EF5350", alpha=0.8, edgecolor="white")
        ax.bar(x + width/2, nonpain_mean, width, yerr=nonpain_std, capsize=3,
               label="Non-Pain", color="#2196F3", alpha=0.8, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(PAIN_NAMES, fontsize=11)
        ax.set_ylabel("Mean Z-Score Deviation from Baseline")
        ax.set_title("SynPAIN: Pain vs Non-Pain AU Deviations",
                     fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "synpain_au_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: synpain_au_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  BAUD × SynPAIN: Synthetic Pain Detection")
    print("  Task: Distinguish pain from non-pain expressions")
    print("=" * 65)

    # Load dataset
    dataset = load_synpain()
    if dataset is None:
        print("❌ Could not load SynPAIN. Install: pip install datasets")
        return

    # Load AU encoder
    model, device = load_model()
    if model is None:
        print("❌ Could not load OpenGraphAU")
        return

    # Run experiment
    baud_results, generic_results, z_score_data = run_experiment(
        dataset, model, device
    )

    if not baud_results:
        print("❌ No results generated. Check data parsing.")
        return

    # Compute metrics
    baud_m = compute_metrics(baud_results, "BAUD")
    gen_m = compute_metrics(generic_results, "Generic")

    print("\n" + "=" * 70)
    print("  SynPAIN RESULTS: Pain vs Non-Pain Expression Detection")
    print("=" * 70)
    print(f"  {'Method':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} "
          f"{'N_pain':>8} {'N_nonpain':>10}")
    print("-" * 70)
    for m in [baud_m, gen_m]:
        print(f"  {m['method']:<20} {m['acc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {m['n_pain']:>8} {m['n_nonpain']:>10}")
    print("=" * 70)

    # Plots
    print("\n📈 Generating plots...")
    plot_results(baud_results, generic_results, z_score_data, RESULTS_DIR)

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "synpain_metrics.txt"), "w") as f:
        f.write("BAUD × SynPAIN Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Task: Pain vs Non-Pain expression detection\n")
        f.write(f"(Harder than pain vs neutral — both deviate from baseline)\n\n")
        for m in [baud_m, gen_m]:
            f.write(f"{m['method']:<20} Acc={m['acc']:.4f} F1={m['f1']:.4f} "
                    f"AUC={m['auc']:.4f}\n")
    print(f"  Saved: synpain_metrics.txt")

    print(f"\n{'=' * 65}")
    print(f"  ✅ SynPAIN EXPERIMENT COMPLETE")
    print(f"{'=' * 65}")
    print(f"  📤 Share console output + plots!")


if __name__ == "__main__":
    main()
