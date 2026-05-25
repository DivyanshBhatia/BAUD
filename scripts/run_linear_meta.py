#!/usr/bin/env python3
"""
BAUD Linear Meta-Learning: Learn optimal deviation weights.

Instead of a 15K-parameter neural network that overfits on 40 subjects,
learn just a WEIGHT VECTOR (41 weights + 1 bias = 42 parameters) that
determines which AU deviations matter most for pain detection.

This is conceptually clean: "we learn from labeled subjects which AUs
are most informative when they deviate from a patient's baseline, then
apply those learned weights to new patients with zero labels."

Usage:
    python scripts/run_linear_meta.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import OneClassSVM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/results"
CACHE_PATH = "/content/results/pemf_extracted_aus.npz"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_AUS = 41
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
ALL_AU_NAMES = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU43", *[f"sub{i}" for i in range(23)]
]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DATA
# ============================================================================

def load_data():
    cached = np.load(CACHE_PATH)
    au_data = {}
    for key in cached.files:
        subj, expr = key[:4], key[5:]
        if subj not in au_data:
            au_data[subj] = {}
        au_data[subj][expr] = cached[key]
    return au_data


def split_subjects(au_data, train_r=0.6, val_r=0.15, seed=SEED):
    subjects = sorted([s for s in au_data if "Neutral" in au_data[s]])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(subjects))
    n_tr = int(len(subjects) * train_r)
    n_val = int(len(subjects) * val_r)
    return ([subjects[i] for i in idx[:n_tr]],
            [subjects[i] for i in idx[n_tr:n_tr + n_val]],
            [subjects[i] for i in idx[n_tr + n_val:]])


# ============================================================================
# LINEAR BAUD MODEL (42 parameters!)
# ============================================================================

class BAUDLinear(nn.Module):
    """
    Learns a single weight vector over AU deviations.

    pain_score = sigmoid(w · z_positive + bias)

    where w is a learned softmax-normalized weight vector (41 dims)
    and z_positive are per-AU z-scores clamped to positive.

    Total parameters: 41 (weights) + 1 (bias) = 42
    """
    def __init__(self, num_aus=NUM_AUS):
        super().__init__()
        # Raw logits → softmax gives normalized weights
        self.weight_logits = nn.Parameter(torch.zeros(num_aus))
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

        # Initialize: slightly favor pain-related AUs
        with torch.no_grad():
            for idx in PAIN_IDX:
                self.weight_logits[idx] = 1.0

    @property
    def weights(self):
        """Normalized deviation weights (sum to 1)."""
        return torch.softmax(self.weight_logits, dim=0)

    def forward(self, z_positive):
        """
        z_positive: (..., num_aus) — positive z-scores
        Returns: pain_scores in [0, 1]
        """
        w = self.weights
        raw = (z_positive * w).sum(dim=-1) * self.scale + self.bias
        return torch.sigmoid(raw)

    def score_subject(self, neutral_aus, pain_aus, eps=1e-4):
        """Calibrate on neutral, score pain. Both (seq, aus) tensors."""
        mean = neutral_aus.mean(0)
        std = neutral_aus.std(0).clamp(min=eps)
        z_pain = ((pain_aus - mean) / std).clamp(min=0)
        return self.forward(z_pain)

    def score_neutral(self, neutral_aus, eps=1e-4):
        """Score neutral against itself."""
        mean = neutral_aus.mean(0)
        std = neutral_aus.std(0).clamp(min=eps)
        z = ((neutral_aus - mean) / std).clamp(min=0)
        return self.forward(z)


# ============================================================================
# ALSO TRY: 2-Layer model (small but nonlinear)
# ============================================================================

class BAUDSmallMLP(nn.Module):
    """
    Tiny MLP: 41 → 16 → 1 = 690 parameters.
    Still small enough to not overfit on 40 subjects.
    """
    def __init__(self, num_aus=NUM_AUS, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_aus, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_positive):
        return torch.sigmoid(self.net(z_positive)).squeeze(-1)

    def score_subject(self, neutral_aus, pain_aus, eps=1e-4):
        mean = neutral_aus.mean(0)
        std = neutral_aus.std(0).clamp(min=eps)
        z = ((pain_aus - mean) / std).clamp(min=0)
        return self.forward(z)

    def score_neutral(self, neutral_aus, eps=1e-4):
        mean = neutral_aus.mean(0)
        std = neutral_aus.std(0).clamp(min=eps)
        z = ((neutral_aus - mean) / std).clamp(min=0)
        return self.forward(z)


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, au_data, train_subj, val_subj, name="Model",
                lr=0.01, epochs=300, patience=30, l2_reg=0.01):
    """Train with per-subject episodes."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses = []
    val_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n = 0

        rng = np.random.RandomState(SEED + epoch)
        order = rng.permutation(len(train_subj))

        for i in order:
            subj = train_subj[i]
            neutral = au_data[subj].get("Neutral")
            if neutral is None:
                continue

            nt = torch.tensor(neutral, dtype=torch.float32).to(DEVICE)

            # Score neutral (target = 0)
            ns = model.score_neutral(nt)

            # Score pain (target = 1)
            pain_scores = []
            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                pain = au_data[subj].get(expr)
                if pain is None:
                    continue
                pt = torch.tensor(pain, dtype=torch.float32).to(DEVICE)
                ps = model.score_subject(nt, pt)
                pain_scores.append(ps)

            if not pain_scores:
                continue

            all_pain = torch.cat(pain_scores)

            # BCE loss
            all_s = torch.cat([all_pain, ns])
            all_t = torch.cat([torch.ones_like(all_pain),
                               torch.zeros_like(ns)])
            loss = nn.functional.binary_cross_entropy(all_s, all_t)

            # Ranking margin loss
            margin_loss = torch.relu(ns.mean() - all_pain.mean() + 0.3)
            loss = loss + 0.3 * margin_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n, 1)
        train_losses.append(avg_loss)

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_m = eval_subjects(model, au_data, val_subj)
            val_history.append({"epoch": epoch + 1, **val_m})

            if val_m["auc"] > best_val_auc:
                best_val_auc = val_m["auc"]
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 10

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  [{name}] Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                      f"Val F1: {val_m['f1']:.4f} | Val AUC: {val_m['auc']:.4f}")

            if no_improve >= patience:
                print(f"  [{name}] Early stopping at epoch {epoch+1}")
                break

    # Load best
    if best_state:
        model.load_state_dict(best_state)
    print(f"  [{name}] Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    return train_losses, val_history


def eval_subjects(model, au_data, subjects):
    """Evaluate model on a set of subjects."""
    model.eval()
    scores, truths = [], []

    with torch.no_grad():
        for subj in subjects:
            neutral = au_data[subj].get("Neutral")
            if neutral is None:
                continue
            nt = torch.tensor(neutral, dtype=torch.float32).to(DEVICE)

            # Neutral → 0
            ns = model.score_neutral(nt)
            scores.append(float(ns.mean()))
            truths.append(0)

            # Pain → 1
            for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
                pain = au_data[subj].get(expr)
                if pain is None:
                    continue
                pt = torch.tensor(pain, dtype=torch.float32).to(DEVICE)
                ps = model.score_subject(nt, pt)
                scores.append(float(ps.mean()))
                truths.append(1)

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

    return {"acc": accuracy_score(truths, preds), "f1": best_f1,
            "auc": auc, "thresh": best_t}


# ============================================================================
# BASELINES
# ============================================================================

def run_baselines(au_data, subjects):
    methods = {}
    for subj in subjects:
        neutral = au_data[subj].get("Neutral")
        if neutral is None:
            continue

        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)
        w = np.ones(NUM_AUS)
        for idx in PAIN_IDX:
            w[idx] = 3.0
        w /= w.sum()

        feat_b = neutral[:, PAIN_IDX]
        mah_mean = np.mean(feat_b, 0)
        mah_cov = np.cov(feat_b, rowvar=False) + np.eye(len(PAIN_IDX)) * 1e-4
        mah_inv = np.linalg.inv(mah_cov)

        try:
            ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
            ocsvm.fit(feat_b)
        except:
            ocsvm = None

        all_exprs = [("Neutral", 0)] + [(e, 1) for e in
                     ["Algometer Pain", "Laser Pain", "Posed Pain"]]

        for expr, label in all_exprs:
            aus = au_data[subj].get(expr)
            if aus is None:
                continue

            # BAUD-Prior
            z = np.maximum((aus - mean_b) / std_b, 0)
            raw = np.array([np.dot(w, f) for f in z])
            baud_s = float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))

            # Generic
            gen_s = float(np.mean([np.mean(f[PAIN_IDX]) for f in aus]))

            # PSPI
            pspi_vals = [min((f[2]+max(f[4],f[5])+max(f[6],f[7])+(f[17] if len(f)>17 else 0))/2, 1) for f in aus]
            pspi_s = float(np.mean(pspi_vals))

            # OC-SVM
            if ocsvm:
                raw_svm = ocsvm.decision_function(aus[:, PAIN_IDX])
                svm_s = float(np.mean([1.0/(1.0+np.exp(r)) for r in raw_svm]))
            else:
                svm_s = 0.5

            # Mahalanobis
            mah_s_list = []
            for f in aus[:, PAIN_IDX]:
                diff = f - mah_mean
                d = float(np.sqrt(diff @ mah_inv @ diff))
                mah_s_list.append(1.0 / (1.0 + np.exp(-d + 3.0)))
            mah_s = float(np.mean(mah_s_list))

            for name, score in [("BAUD-Prior", baud_s), ("Generic", gen_s),
                                ("PSPI", pspi_s), ("One-Class SVM", svm_s),
                                ("Mahalanobis", mah_s)]:
                if name not in methods:
                    methods[name] = {"scores": [], "truths": []}
                methods[name]["scores"].append(score)
                methods[name]["truths"].append(label)

    results = {}
    for name, data in methods.items():
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(data["truths"], [1 if s > t else 0 for s in data["scores"]], zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        preds = [1 if s > best_t else 0 for s in data["scores"]]
        try:
            auc = roc_auc_score(data["truths"], data["scores"])
        except:
            auc = 0.0
        results[name] = {"acc": accuracy_score(data["truths"], preds),
                         "f1": best_f1, "auc": auc}
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_learned_weights(model, save_path):
    """Visualize which AUs the model learned to weight highest."""
    if not hasattr(model, 'weights'):
        return

    w = model.weights.detach().cpu().numpy()

    # Show top 18 AUs (the named ones)
    n_show = min(18, len(w))
    indices = np.argsort(w)[::-1][:n_show]
    names = [ALL_AU_NAMES[i] if i < len(ALL_AU_NAMES) else f"AU_{i}" for i in indices]
    values = w[indices]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#EF5350" if i in PAIN_IDX else "#2196F3" for i in indices]
    ax.bar(range(n_show), values, color=colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Learned Weight")
    ax.set_title("BAUD-Linear: Learned AU Deviation Weights\n"
                 "(Red = known pain AUs, Blue = other AUs)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(all_results, save_path):
    methods = list(all_results.keys())
    f1s = [all_results[m]["f1"] for m in methods]
    aucs = [all_results[m]["auc"] for m in methods]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(methods))
    width = 0.35
    b1 = ax.bar(x - width/2, f1s, width, label="F1", color="#2196F3", alpha=0.8)
    b2 = ax.bar(x + width/2, aucs, width, label="AUC", color="#FF9800", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Complete Method Comparison — PEMF Test Subjects",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.4, 1.08)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training(losses_dict, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Linear (42 params)": "#2196F3", "SmallMLP (690 params)": "#FF9800"}
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, color=colors.get(name, "#666"),
                linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Meta-Training Convergence", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("  BAUD Lightweight Meta-Learning")
    print("  Linear (42 params) + SmallMLP (690 params)")
    print("=" * 65)

    au_data = load_data()
    train_s, val_s, test_s = split_subjects(au_data)
    print(f"  Subjects: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    # ── Train Linear Model ──
    print(f"\n{'─' * 65}")
    print(f"  Training BAUD-Linear (42 parameters)")
    print(f"{'─' * 65}")

    linear_model = BAUDLinear().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in linear_model.parameters())}")

    linear_losses, linear_val = train_model(
        linear_model, au_data, train_s, val_s,
        name="Linear", lr=0.02, epochs=500, patience=50, l2_reg=0.001
    )

    # ── Train SmallMLP Model ──
    print(f"\n{'─' * 65}")
    print(f"  Training BAUD-SmallMLP (690 parameters)")
    print(f"{'─' * 65}")

    mlp_model = BAUDSmallMLP(hidden=16).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in mlp_model.parameters())}")

    mlp_losses, mlp_val = train_model(
        mlp_model, au_data, train_s, val_s,
        name="SmallMLP", lr=0.01, epochs=500, patience=50, l2_reg=0.005
    )

    # ── Evaluate on Test Subjects ──
    print(f"\n{'─' * 65}")
    print(f"  EVALUATION ON TEST SUBJECTS ({len(test_s)} unseen subjects)")
    print(f"{'─' * 65}")

    linear_test = eval_subjects(linear_model, au_data, test_s)
    mlp_test = eval_subjects(mlp_model, au_data, test_s)
    baseline_results = run_baselines(au_data, test_s)

    all_results = {}
    all_results["BAUD-Linear (Ours)"] = linear_test
    all_results["BAUD-SmallMLP (Ours)"] = mlp_test
    all_results.update(baseline_results)

    # Print final table
    print("\n" + "=" * 80)
    print("  FINAL RESULTS — Test Subjects (Never Seen During Training)")
    print("=" * 80)
    print(f"  {'Method':<28} {'Acc':>8} {'F1':>8} {'AUC':>8}  "
          f"{'Params':>8} {'Personal':>10}")
    print("-" * 80)

    param_info = {
        "BAUD-Linear (Ours)": "42",
        "BAUD-SmallMLP (Ours)": "690",
        "BAUD-Prior": "0 (hand)",
        "Generic": "0",
        "PSPI": "0",
        "One-Class SVM": "auto",
        "Mahalanobis": "0 (stat)",
    }

    for method, m in all_results.items():
        params = param_info.get(method, "?")
        personal = "Yes" if method not in ["Generic", "PSPI"] else "No"
        print(f"  {method:<28} {m['acc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f}  {params:>8} {personal:>10}")
    print("=" * 80)

    # ── Visualizations ──
    print("\n📈 Generating plots...")

    plot_learned_weights(
        linear_model,
        os.path.join(RESULTS_DIR, "learned_au_weights.png")
    )
    plot_comparison(
        all_results,
        os.path.join(RESULTS_DIR, "linear_meta_comparison.png")
    )
    plot_training(
        {"Linear (42 params)": linear_losses,
         "SmallMLP (690 params)": mlp_losses},
        os.path.join(RESULTS_DIR, "linear_meta_training.png")
    )

    # Print learned weights
    if hasattr(linear_model, 'weights'):
        w = linear_model.weights.detach().cpu().numpy()
        print("\n  Learned AU Deviation Weights (top 10):")
        print("  " + "-" * 40)
        top_idx = np.argsort(w)[::-1][:10]
        for i in top_idx:
            name = ALL_AU_NAMES[i] if i < len(ALL_AU_NAMES) else f"AU_{i}"
            pain_marker = " ★ PAIN" if i in PAIN_IDX else ""
            print(f"    {name:>6s}: {w[i]:.4f}{pain_marker}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "linear_meta_results.txt")
    with open(results_path, "w") as f:
        f.write("BAUD Lightweight Meta-Learning Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Method':<28} {'Acc':>8} {'F1':>8} {'AUC':>8}\n")
        f.write("-" * 80 + "\n")
        for method, m in all_results.items():
            f.write(f"{method:<28} {m['acc']:>8.4f} {m['f1']:>8.4f} "
                    f"{m['auc']:>8.4f}\n")
    print(f"  Saved: {results_path}")

    # Save model
    torch.save(linear_model.state_dict(),
               os.path.join(RESULTS_DIR, "baud_linear_best.pth"))
    torch.save(mlp_model.state_dict(),
               os.path.join(RESULTS_DIR, "baud_mlp_best.pth"))

    print(f"\n{'=' * 65}")
    print(f"  ✅ LIGHTWEIGHT META-LEARNING COMPLETE")
    print(f"{'=' * 65}")
    print(f"\n  Results: {RESULTS_DIR}/")
    print(f"  ├── learned_au_weights.png     ← which AUs matter for pain")
    print(f"  ├── linear_meta_comparison.png  ← method comparison bars")
    print(f"  ├── linear_meta_training.png    ← training curves")
    print(f"  ├── linear_meta_results.txt     ← numbers")
    print(f"  ├── baud_linear_best.pth        ← trained linear model")
    print(f"  └── baud_mlp_best.pth           ← trained MLP model")
    print(f"\n  📤 Share results to review!")


if __name__ == "__main__":
    main()
