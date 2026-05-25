#!/usr/bin/env python3
"""
BAUD Meta-Learning: Learn deviation weights across subjects.

This is the core technical contribution:
- Meta-train: learn which AU deviations predict pain (using labeled subjects)
- Meta-test: apply learned weights to NEW subjects (zero labels per patient)

The learned weights should outperform:
- Hand-crafted prior weights (current BAUD)
- Unweighted statistical methods (Mahalanobis)
- Standard anomaly detection (One-Class SVM, Isolation Forest)

Usage on Colab:
    python scripts/run_meta_learning.py
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import OneClassSVM
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
RESULTS_DIR = "/content/results"
CACHE_PATH = "/content/results/pemf_extracted_aus.npz"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_AUS = 41
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta-learning hyperparams
META_LR = 5e-4
META_EPOCHS = 150
PATIENCE = 20        # Early stopping patience
LAMBDA_RANK = 0.5
LAMBDA_TEMPORAL = 0.1


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cached_aus(cache_path=CACHE_PATH):
    """Load pre-extracted AUs from the OpenGraphAU cache."""
    print(f"Loading cached AUs from {cache_path}...")
    cached = np.load(cache_path)
    au_data = {}
    for key in cached.files:
        subj = key[:4]
        expr = key[5:]
        if subj not in au_data:
            au_data[subj] = {}
        au_data[subj][expr] = cached[key]
    print(f"  Loaded {len(au_data)} subjects")
    return au_data


def split_subjects(au_data, train_ratio=0.6, val_ratio=0.15, seed=SEED):
    """Split subjects into meta-train / meta-val / meta-test."""
    subjects = sorted([s for s in au_data.keys() if "Neutral" in au_data[s]])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(subjects))

    n_train = int(len(subjects) * train_ratio)
    n_val = int(len(subjects) * val_ratio)

    train = [subjects[i] for i in idx[:n_train]]
    val = [subjects[i] for i in idx[n_train:n_train + n_val]]
    test = [subjects[i] for i in idx[n_train + n_val:]]

    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ============================================================================
# MODEL (inline for self-contained script)
# ============================================================================

class DeviationWeightNet(nn.Module):
    def __init__(self, num_aus=NUM_AUS, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_aus, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z)


class TemporalGRU(nn.Module):
    def __init__(self, num_aus=NUM_AUS, hidden=32):
        super().__init__()
        self.gru = nn.GRU(num_aus, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, num_aus)

    def forward(self, z_seq):
        out, _ = self.gru(z_seq)
        return self.proj(out)


class BAUDMeta(nn.Module):
    """Meta-learnable BAUD model."""

    def __init__(self, num_aus=NUM_AUS, hidden=64, gru_hidden=32):
        super().__init__()
        self.num_aus = num_aus
        self.temporal = TemporalGRU(num_aus, gru_hidden)
        self.weight_net = DeviationWeightNet(num_aus, hidden)

    def forward(self, z_scores):
        """z_scores: (batch, seq, aus) → pain_scores: (batch, seq)"""
        agg = self.temporal(z_scores)
        B, T, N = agg.shape
        logits = self.weight_net(agg.reshape(B * T, N))
        return torch.sigmoid(logits).reshape(B, T)

    def score_subject(self, neutral_aus, pain_aus, eps=1e-4):
        """Calibrate on neutral, score pain. Both are (seq, aus) tensors."""
        mean = neutral_aus.mean(0, keepdim=True)
        std = neutral_aus.std(0, keepdim=True).clamp(min=eps)
        z = ((pain_aus - mean) / std).clamp(min=0).unsqueeze(0)
        return self.forward(z).squeeze(0)

    def score_neutral(self, neutral_aus, eps=1e-4):
        """Score neutral against itself (should be low)."""
        mean = neutral_aus.mean(0, keepdim=True)
        std = neutral_aus.std(0, keepdim=True).clamp(min=eps)
        z = ((neutral_aus - mean) / std).clamp(min=0).unsqueeze(0)
        return self.forward(z).squeeze(0)


# ============================================================================
# META-TRAINING LOOP
# ============================================================================

def meta_train_epoch(model, optimizer, au_data, train_subjects, device):
    """One epoch of meta-training: iterate over training subjects."""
    model.train()
    epoch_losses = []

    for subj in train_subjects:
        neutral = au_data[subj].get("Neutral")
        if neutral is None:
            continue

        neutral_t = torch.tensor(neutral, dtype=torch.float32).to(device)

        # Score neutral frames (target: 0)
        neutral_scores = model.score_neutral(neutral_t)
        neutral_target = torch.zeros_like(neutral_scores)

        # Score all pain expressions (target: 1)
        pain_scores_list = []
        for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            pain = au_data[subj].get(expr)
            if pain is None:
                continue
            pain_t = torch.tensor(pain, dtype=torch.float32).to(device)
            ps = model.score_subject(neutral_t, pain_t)
            pain_scores_list.append(ps)

        if not pain_scores_list:
            continue

        all_pain = torch.cat(pain_scores_list)
        pain_target = torch.ones_like(all_pain)

        # Combined BCE loss
        all_scores = torch.cat([all_pain, neutral_scores])
        all_targets = torch.cat([pain_target, neutral_target])
        l_bce = nn.functional.binary_cross_entropy(all_scores, all_targets)

        # Ranking loss: pain mean > neutral mean + margin
        margin = 0.3
        l_rank = torch.relu(neutral_scores.mean() - all_pain.mean() + margin)

        # Temporal smoothness on pain scores
        l_temp = torch.tensor(0.0, device=device)
        for ps in pain_scores_list:
            if len(ps) > 1:
                l_temp += (ps[1:] - ps[:-1]).pow(2).mean()

        loss = l_bce + LAMBDA_RANK * l_rank + LAMBDA_TEMPORAL * l_temp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses) if epoch_losses else float("inf")


def evaluate(model, au_data, subjects, device):
    """Evaluate on a set of subjects. Returns metrics dict."""
    model.eval()
    all_scores = []
    all_truths = []

    with torch.no_grad():
        for subj in subjects:
            neutral = au_data[subj].get("Neutral")
            if neutral is None:
                continue
            neutral_t = torch.tensor(neutral, dtype=torch.float32).to(device)

            # Neutral → should be low
            ns = model.score_neutral(neutral_t)
            mean_ns = float(ns.mean())
            all_scores.append(mean_ns)
            all_truths.append(0)

            # Pain → should be high
            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                pain = au_data[subj].get(expr)
                if pain is None:
                    continue
                pain_t = torch.tensor(pain, dtype=torch.float32).to(device)
                ps = model.score_subject(neutral_t, pain_t)
                mean_ps = float(ps.mean())
                all_scores.append(mean_ps)
                all_truths.append(1)

    # Find best threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds = [1 if s > t else 0 for s in all_scores]
        f1 = f1_score(all_truths, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    preds = [1 if s > best_thresh else 0 for s in all_scores]
    acc = accuracy_score(all_truths, preds)
    try:
        auc = roc_auc_score(all_truths, all_scores)
    except:
        auc = 0.0

    return {"acc": acc, "f1": best_f1, "auc": auc, "thresh": best_thresh,
            "scores": all_scores, "truths": all_truths}


# ============================================================================
# BASELINES (for comparison)
# ============================================================================

def run_all_baselines(au_data, subjects):
    """Run all baseline methods on given subjects."""
    methods = {}

    for method_name in ["BAUD-Prior", "Generic", "PSPI",
                        "One-Class SVM", "Mahalanobis"]:
        scores, truths = [], []

        for subj in subjects:
            neutral = au_data[subj].get("Neutral")
            if neutral is None:
                continue

            mean_b = np.mean(neutral, axis=0)
            std_b = np.maximum(np.std(neutral, axis=0), 1e-4)

            # Prior weights
            weights = np.ones(NUM_AUS)
            for idx in PAIN_IDX:
                weights[idx] = 3.0
            weights /= weights.sum()

            # Mahalanobis setup
            feat_b = neutral[:, PAIN_IDX]
            mah_mean = np.mean(feat_b, axis=0)
            mah_cov = np.cov(feat_b, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
            mah_inv = np.linalg.inv(mah_cov)

            # OC-SVM setup
            try:
                ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
                ocsvm.fit(feat_b)
            except:
                ocsvm = None

            def score_frames(aus, is_neutral=False):
                results = {}

                # BAUD-Prior
                z = np.maximum((aus - mean_b) / std_b, 0)
                raw = np.array([np.dot(weights, frame) for frame in z])
                baud_s = 1.0 / (1.0 + np.exp(-raw + 2.0))
                results["BAUD-Prior"] = float(np.mean(baud_s))

                # Generic
                results["Generic"] = float(np.mean([np.mean(f[PAIN_IDX])
                                                     for f in aus]))
                # PSPI
                pspi_vals = []
                for f in aus:
                    p = f[2] + max(f[4], f[5]) + max(f[6], f[7])
                    p += f[17] if len(f) > 17 else 0
                    pspi_vals.append(min(p / 2.0, 1.0))
                results["PSPI"] = float(np.mean(pspi_vals))

                # OC-SVM
                if ocsvm is not None:
                    raw_svm = ocsvm.decision_function(aus[:, PAIN_IDX])
                    results["One-Class SVM"] = float(
                        np.mean([1.0 / (1.0 + np.exp(r)) for r in raw_svm]))
                else:
                    results["One-Class SVM"] = 0.5

                # Mahalanobis
                mah_scores = []
                for f in aus[:, PAIN_IDX]:
                    diff = f - mah_mean
                    d = float(np.sqrt(diff @ mah_inv @ diff))
                    mah_scores.append(1.0 / (1.0 + np.exp(-d + 3.0)))
                results["Mahalanobis"] = float(np.mean(mah_scores))

                return results

            # Score neutral
            for name, score in score_frames(neutral, True).items():
                if name not in methods:
                    methods[name] = {"scores": [], "truths": []}
                methods[name]["scores"].append(score)
                methods[name]["truths"].append(0)

            # Score pain
            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                pain = au_data[subj].get(expr)
                if pain is None:
                    continue
                for name, score in score_frames(pain).items():
                    methods[name]["scores"].append(score)
                    methods[name]["truths"].append(1)

    # Compute metrics
    results = {}
    for name, data in methods.items():
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            preds = [1 if s > t else 0 for s in data["scores"]]
            f1 = f1_score(data["truths"], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        preds = [1 if s > best_t else 0 for s in data["scores"]]
        try:
            auc = roc_auc_score(data["truths"], data["scores"])
        except:
            auc = 0.0
        results[name] = {
            "acc": accuracy_score(data["truths"], preds),
            "f1": best_f1, "auc": auc,
        }
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curve(train_losses, val_metrics, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, color="#2196F3", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Meta-Training Loss", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    epochs = [v["epoch"] for v in val_metrics]
    ax2.plot(epochs, [v["f1"] for v in val_metrics], "o-",
             color="#4CAF50", label="F1", linewidth=1.5)
    ax2.plot(epochs, [v["auc"] for v in val_metrics], "s-",
             color="#FF9800", label="AUC", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_final_comparison(all_results, save_path):
    methods = list(all_results.keys())
    f1s = [all_results[m]["f1"] for m in methods]
    aucs = [all_results[m]["auc"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, f1s, width, label="F1", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, aucs, width, label="AUC", color="#FF9800", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Complete Method Comparison on PEMF (Test Subjects)",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  BAUD Meta-Learning: Learning Deviation Weights")
    print("=" * 65)

    # Load data
    au_data = load_cached_aus()
    train_subj, val_subj, test_subj = split_subjects(au_data)

    # Initialize model
    model = BAUDMeta(num_aus=NUM_AUS, hidden=64, gru_hidden=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=META_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )

    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")

    # ── Meta-Training ──
    print(f"\n{'─' * 65}")
    print(f"  META-TRAINING ({META_EPOCHS} epochs, {len(train_subj)} subjects)")
    print(f"{'─' * 65}")

    train_losses = []
    val_metrics_history = []
    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0

    for epoch in range(META_EPOCHS):
        # Shuffle training order each epoch
        rng = np.random.RandomState(SEED + epoch)
        shuffled = [train_subj[i] for i in rng.permutation(len(train_subj))]

        loss = meta_train_epoch(model, optimizer, au_data, shuffled, DEVICE)
        train_losses.append(loss)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_result = evaluate(model, au_data, val_subj, DEVICE)
            val_metrics_history.append({"epoch": epoch + 1, **val_result})
            scheduler.step(val_result["f1"])

            print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Val Acc: {val_result['acc']:.4f} | "
                  f"Val F1: {val_result['f1']:.4f} | "
                  f"Val AUC: {val_result['auc']:.4f}")

            if val_result["f1"] > best_val_f1:
                best_val_f1 = val_result["f1"]
                best_epoch = epoch + 1
                no_improve = 0
                # Save best model
                torch.save(model.state_dict(),
                           os.path.join(RESULTS_DIR, "baud_meta_best.pth"))
            else:
                no_improve += 5

            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n  Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "baud_meta_best.pth"),
                    map_location=DEVICE)
    )

    # ── Evaluation on Test Subjects ──
    print(f"\n{'─' * 65}")
    print(f"  EVALUATION ON TEST SUBJECTS ({len(test_subj)} subjects)")
    print(f"  These subjects were NEVER seen during training")
    print(f"{'─' * 65}")

    # BAUD-Meta (ours)
    test_result = evaluate(model, au_data, test_subj, DEVICE)

    # All baselines on same test subjects
    baseline_results = run_all_baselines(au_data, test_subj)

    # Combine
    all_results = {
        "BAUD-Meta (Ours)": {
            "acc": test_result["acc"],
            "f1": test_result["f1"],
            "auc": test_result["auc"],
        }
    }
    all_results.update(baseline_results)

    # Print final table
    print("\n" + "=" * 78)
    print("  FINAL RESULTS: Test Subjects (Never Seen During Training)")
    print("=" * 78)
    print(f"  {'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}  "
          f"{'Personalized':>12} {'Labels/Patient':>14}")
    print("-" * 78)

    method_info = {
        "BAUD-Meta (Ours)": ("Yes", "Zero"),
        "BAUD-Prior": ("Yes", "Zero"),
        "Generic": ("No", "N/A"),
        "PSPI": ("No", "N/A"),
        "One-Class SVM": ("Yes", "Zero"),
        "Mahalanobis": ("Yes", "Zero"),
    }

    for method, metrics in all_results.items():
        personal, labels = method_info.get(method, ("?", "?"))
        print(f"  {method:<25} {metrics['acc']:>8.4f} {metrics['f1']:>8.4f} "
              f"{metrics['auc']:>8.4f}  {personal:>12} {labels:>14}")
    print("=" * 78)

    # ── Visualizations ──
    print("\n📈 Generating plots...")
    plot_training_curve(
        train_losses, val_metrics_history,
        os.path.join(RESULTS_DIR, "meta_training_curve.png")
    )
    plot_final_comparison(
        all_results,
        os.path.join(RESULTS_DIR, "meta_final_comparison.png")
    )

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "meta_learning_results.txt")
    with open(metrics_path, "w") as f:
        f.write("BAUD Meta-Learning Results\n")
        f.write("=" * 78 + "\n")
        f.write(f"{'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}\n")
        f.write("-" * 78 + "\n")
        for method, m in all_results.items():
            f.write(f"{method:<25} {m['acc']:>8.4f} {m['f1']:>8.4f} "
                    f"{m['auc']:>8.4f}\n")
        f.write("=" * 78 + "\n")
        f.write(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}\n")
        f.write(f"Train subjects: {len(train_subj)}\n")
        f.write(f"Val subjects: {len(val_subj)}\n")
        f.write(f"Test subjects: {len(test_subj)}\n")
    print(f"  Saved: {metrics_path}")

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  ✅ META-LEARNING EXPERIMENT COMPLETE")
    print(f"{'=' * 65}")
    print(f"\n  Results in: {RESULTS_DIR}/")
    print(f"  ├── meta_training_curve.png")
    print(f"  ├── meta_final_comparison.png")
    print(f"  ├── meta_learning_results.txt")
    print(f"  └── baud_meta_best.pth (trained model)")
    print(f"\n  📤 Share these files to review results!")


if __name__ == "__main__":
    main()
