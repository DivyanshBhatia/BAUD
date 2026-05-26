#!/usr/bin/env python3
"""
Cross-Dataset Transfer: Do BAUD's learned weights generalize?

Train deviation weights on PEMF → Test on UNBC (and vice versa).
If weights transfer, it means BAUD learns universal pain-AU patterns,
not dataset-specific artifacts.

Run on Colab:
    python scripts/run_cross_dataset.py
"""
import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import OneClassSVM
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/results"
PEMF_CACHE = "/content/results/pemf_extracted_aus.npz"
UNBC_CACHE = "/content/results/unbc_extracted_aus.npz"
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIN_IDX = [2, 4, 5, 6, 7, 17]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# DATA
# ============================================================================

def load_pemf():
    cached = np.load(PEMF_CACHE)
    data = {}
    for key in cached.files:
        subj, expr = key[:4], key[5:]
        if subj not in data:
            data[subj] = {}
        data[subj][expr] = cached[key]
    return data

def load_unbc():
    cached = np.load(UNBC_CACHE)
    data = {}
    for key in cached.files:
        if "_pspi" in key:
            continue
        subj = key.split("_")[0]
        label = key.split("_", 1)[1]
        if subj not in data:
            data[subj] = {}
        data[subj][label] = cached[key]
    return data

def get_neutral_pain(data, dataset):
    """Get neutral and pain AU arrays per subject."""
    subjects = {}
    for subj, exprs in data.items():
        if dataset == "pemf":
            neutral = exprs.get("Neutral")
            pain_list = [exprs[e] for e in ["Algometer Pain", "Laser Pain", "Posed Pain"] if e in exprs]
        else:  # unbc
            neutral = exprs.get("neutral")
            pain_list = [exprs[e] for e in exprs if e != "neutral"]

        if neutral is not None and pain_list:
            subjects[subj] = {"neutral": neutral, "pain": np.concatenate(pain_list)}
    return subjects


# ============================================================================
# BAUD-LINEAR MODEL
# ============================================================================

class BAUDLinear(nn.Module):
    def __init__(self, num_aus=41):
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(num_aus))
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            for idx in PAIN_IDX:
                self.weight_logits[idx] = 1.0

    @property
    def weights(self):
        return torch.softmax(self.weight_logits, dim=0)

    def forward(self, z_positive):
        w = self.weights
        raw = (z_positive * w).sum(dim=-1) * self.scale + self.bias
        return torch.sigmoid(raw)


def train_baud_on_dataset(subjects, epochs=500, lr=0.02, patience=50):
    """Train BAUD-Linear on a set of subjects."""
    model = BAUDLinear().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Split into train/val (80/20)
    subj_ids = sorted(subjects.keys())
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(subj_ids))
    n_train = int(len(subj_ids) * 0.8)
    train_ids = [subj_ids[i] for i in idx[:n_train]]
    val_ids = [subj_ids[i] for i in idx[n_train:]]

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        rng_ep = np.random.RandomState(SEED + epoch)
        order = rng_ep.permutation(len(train_ids))

        epoch_loss = 0
        for i in order:
            subj = train_ids[i]
            neutral = subjects[subj]["neutral"]
            pain = subjects[subj]["pain"]

            nt = torch.tensor(neutral, dtype=torch.float32).to(DEVICE)
            pt = torch.tensor(pain, dtype=torch.float32).to(DEVICE)

            # Z-scores
            mean_b = nt.mean(0)
            std_b = nt.std(0).clamp(min=1e-4)
            z_n = ((nt - mean_b) / std_b).clamp(min=0)
            z_p = ((pt - mean_b) / std_b).clamp(min=0)

            s_n = model(z_n)
            s_p = model(z_p)

            # BCE
            all_s = torch.cat([s_p, s_n])
            all_t = torch.cat([torch.ones_like(s_p), torch.zeros_like(s_n)])
            loss = nn.functional.binary_cross_entropy(all_s, all_t)
            loss += 0.3 * torch.relu(s_n.mean() - s_p.mean() + 0.3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validate
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_scores, val_truths = [], []
            with torch.no_grad():
                for subj in val_ids:
                    neutral = subjects[subj]["neutral"]
                    pain = subjects[subj]["pain"]
                    nt = torch.tensor(neutral, dtype=torch.float32).to(DEVICE)
                    pt = torch.tensor(pain, dtype=torch.float32).to(DEVICE)
                    mean_b = nt.mean(0)
                    std_b = nt.std(0).clamp(min=1e-4)

                    z_n = ((nt - mean_b) / std_b).clamp(min=0)
                    z_p = ((pt - mean_b) / std_b).clamp(min=0)

                    val_scores.append(float(model(z_n).mean()))
                    val_truths.append(0)
                    val_scores.append(float(model(z_p).mean()))
                    val_truths.append(1)

            try:
                auc = roc_auc_score(val_truths, val_scores)
            except:
                auc = 0.5

            if auc > best_val_auc:
                best_val_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 10
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_baud(model, subjects):
    """Evaluate trained BAUD model on a set of subjects."""
    model.eval()
    scores, truths = [], []
    with torch.no_grad():
        for subj in sorted(subjects.keys()):
            neutral = subjects[subj]["neutral"]
            pain = subjects[subj]["pain"]
            nt = torch.tensor(neutral, dtype=torch.float32).to(DEVICE)
            pt = torch.tensor(pain, dtype=torch.float32).to(DEVICE)
            mean_b = nt.mean(0)
            std_b = nt.std(0).clamp(min=1e-4)

            z_n = ((nt - mean_b) / std_b).clamp(min=0)
            z_p = ((pt - mean_b) / std_b).clamp(min=0)

            scores.append(float(model(z_n).mean()))
            truths.append(0)
            scores.append(float(model(z_p).mean()))
            truths.append(1)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    preds = [1 if s > best_t else 0 for s in scores]
    try:
        auc = roc_auc_score(truths, scores)
    except:
        auc = 0
    return {"acc": accuracy_score(truths, preds), "f1": best_f1, "auc": auc}


def evaluate_prior(subjects):
    """BAUD with hand-crafted prior weights (no training)."""
    scores, truths = [], []
    for subj in sorted(subjects.keys()):
        neutral = subjects[subj]["neutral"]
        pain = subjects[subj]["pain"]
        mean_b = np.mean(neutral, 0)
        std_b = np.maximum(np.std(neutral, 0), 1e-4)
        w = np.ones(neutral.shape[1])
        for i in PAIN_IDX: w[i] = 3.0
        w /= w.sum()

        for aus, label in [(neutral, 0), (pain, 1)]:
            z = np.maximum((aus - mean_b) / std_b, 0)
            raw = np.array([np.dot(w, f) for f in z])
            s = float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))
            scores.append(s)
            truths.append(label)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    preds = [1 if s > best_t else 0 for s in scores]
    try: auc = roc_auc_score(truths, scores)
    except: auc = 0
    return {"acc": accuracy_score(truths, preds), "f1": best_f1, "auc": auc}


def evaluate_generic(subjects):
    """Generic baseline (no personalization)."""
    scores, truths = [], []
    for subj in sorted(subjects.keys()):
        for aus, label in [(subjects[subj]["neutral"], 0), (subjects[subj]["pain"], 1)]:
            scores.append(float(np.mean([np.mean(f[PAIN_IDX]) for f in aus])))
            truths.append(label)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    preds = [1 if s > best_t else 0 for s in scores]
    try: auc = roc_auc_score(truths, scores)
    except: auc = 0
    return {"acc": accuracy_score(truths, preds), "f1": best_f1, "auc": auc}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  Cross-Dataset Transfer Experiment")
    print("  Do BAUD's learned weights generalize across datasets?")
    print("=" * 70)

    # Load both datasets
    print("\n📂 Loading datasets...")
    pemf_raw = load_pemf()
    unbc_raw = load_unbc()
    pemf = get_neutral_pain(pemf_raw, "pemf")
    unbc = get_neutral_pain(unbc_raw, "unbc")
    print(f"  PEMF: {len(pemf)} subjects")
    print(f"  UNBC: {len(unbc)} subjects")

    # ── Experiment 1: Train on PEMF → Test on UNBC ──
    print(f"\n{'─'*70}")
    print(f"  Train on PEMF ({len(pemf)} subjects) → Test on UNBC ({len(unbc)} subjects)")
    print(f"{'─'*70}")

    pemf_model = train_baud_on_dataset(pemf)
    print("  ✅ BAUD-Linear trained on PEMF")

    # Evaluate on both
    pemf_on_pemf = evaluate_baud(pemf_model, pemf)
    pemf_on_unbc = evaluate_baud(pemf_model, unbc)

    print(f"  BAUD-Linear (PEMF→PEMF):  AUC={pemf_on_pemf['auc']:.4f}  F1={pemf_on_pemf['f1']:.4f}")
    print(f"  BAUD-Linear (PEMF→UNBC):  AUC={pemf_on_unbc['auc']:.4f}  F1={pemf_on_unbc['f1']:.4f}")

    # ── Experiment 2: Train on UNBC → Test on PEMF ──
    print(f"\n{'─'*70}")
    print(f"  Train on UNBC ({len(unbc)} subjects) → Test on PEMF ({len(pemf)} subjects)")
    print(f"{'─'*70}")

    unbc_model = train_baud_on_dataset(unbc)
    print("  ✅ BAUD-Linear trained on UNBC")

    unbc_on_unbc = evaluate_baud(unbc_model, unbc)
    unbc_on_pemf = evaluate_baud(unbc_model, pemf)

    print(f"  BAUD-Linear (UNBC→UNBC):  AUC={unbc_on_unbc['auc']:.4f}  F1={unbc_on_unbc['f1']:.4f}")
    print(f"  BAUD-Linear (UNBC→PEMF):  AUC={unbc_on_pemf['auc']:.4f}  F1={unbc_on_pemf['f1']:.4f}")

    # ── Baselines for comparison ──
    print(f"\n{'─'*70}")
    print(f"  Baselines (no cross-dataset training)")
    print(f"{'─'*70}")

    prior_pemf = evaluate_prior(pemf)
    prior_unbc = evaluate_prior(unbc)
    gen_pemf = evaluate_generic(pemf)
    gen_unbc = evaluate_generic(unbc)

    print(f"  BAUD-Prior on PEMF:       AUC={prior_pemf['auc']:.4f}")
    print(f"  BAUD-Prior on UNBC:       AUC={prior_unbc['auc']:.4f}")
    print(f"  Generic on PEMF:          AUC={gen_pemf['auc']:.4f}")
    print(f"  Generic on UNBC:          AUC={gen_unbc['auc']:.4f}")

    # ── Print learned weights comparison ──
    print(f"\n{'─'*70}")
    print(f"  Learned Weight Comparison")
    print(f"{'─'*70}")

    pemf_w = pemf_model.weights.detach().cpu().numpy()
    unbc_w = unbc_model.weights.detach().cpu().numpy()

    au_names = ["AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10",
                "AU12","AU14","AU15","AU17","AU20","AU23","AU24",
                "AU25","AU26","AU43"] + [f"s{i}" for i in range(23)]

    print(f"  {'AU':<8} {'PEMF-trained':>14} {'UNBC-trained':>14} {'Pain AU?':>10}")
    print(f"  {'-'*50}")
    top_pemf = np.argsort(pemf_w)[::-1][:10]
    for i in top_pemf:
        name = au_names[i] if i < len(au_names) else f"AU_{i}"
        pain = "★" if i in PAIN_IDX else ""
        print(f"  {name:<8} {pemf_w[i]:>14.4f} {unbc_w[i]:>14.4f} {pain:>10}")

    # Weight correlation
    corr = np.corrcoef(pemf_w, unbc_w)[0, 1]
    print(f"\n  Weight correlation (PEMF vs UNBC): r = {corr:.4f}")

    # ── Summary Table ──
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET TRANSFER RESULTS")
    print(f"{'='*70}")
    print(f"  {'Configuration':<30} {'PEMF AUC':>10} {'UNBC AUC':>10}")
    print(f"  {'-'*55}")
    print(f"  {'BAUD-Linear (train=PEMF)':<30} {pemf_on_pemf['auc']:>10.4f} {pemf_on_unbc['auc']:>10.4f}")
    print(f"  {'BAUD-Linear (train=UNBC)':<30} {unbc_on_pemf['auc']:>10.4f} {unbc_on_unbc['auc']:>10.4f}")
    print(f"  {'BAUD-Prior (no training)':<30} {prior_pemf['auc']:>10.4f} {prior_unbc['auc']:>10.4f}")
    print(f"  {'Generic (no personal.)':<30} {gen_pemf['auc']:>10.4f} {gen_unbc['auc']:>10.4f}")
    print(f"{'='*70}")
    print(f"  Weight correlation: r = {corr:.4f}")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: cross-dataset AUCs
    configs = ["BAUD-Linear\n(train=PEMF)", "BAUD-Linear\n(train=UNBC)",
               "BAUD-Prior\n(no training)", "Generic"]
    pemf_aucs = [pemf_on_pemf["auc"], unbc_on_pemf["auc"], prior_pemf["auc"], gen_pemf["auc"]]
    unbc_aucs = [pemf_on_unbc["auc"], unbc_on_unbc["auc"], prior_unbc["auc"], gen_unbc["auc"]]

    x = np.arange(len(configs))
    ax1.bar(x - 0.175, pemf_aucs, 0.35, label="Test on PEMF", color="#2196F3", alpha=0.8)
    ax1.bar(x + 0.175, unbc_aucs, 0.35, label="Test on UNBC", color="#FF9800", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylabel("AUC")
    ax1.set_title("Cross-Dataset Transfer", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.set_ylim(0.4, 1.08)
    ax1.grid(True, alpha=0.3, axis="y")
    for i, (p, u) in enumerate(zip(pemf_aucs, unbc_aucs)):
        ax1.text(i - 0.175, p + 0.02, f"{p:.3f}", ha="center", fontsize=8)
        ax1.text(i + 0.175, u + 0.02, f"{u:.3f}", ha="center", fontsize=8)

    # Weight comparison scatter
    ax2.scatter(pemf_w, unbc_w, c=["#EF5350" if i in PAIN_IDX else "#2196F3"
                                    for i in range(len(pemf_w))],
                s=80, alpha=0.7, edgecolors="white")
    ax2.set_xlabel("Weight (PEMF-trained)", fontsize=11)
    ax2.set_ylabel("Weight (UNBC-trained)", fontsize=11)
    ax2.set_title(f"Learned Weight Correlation (r={corr:.3f})",
                  fontsize=13, fontweight="bold")
    ax2.plot([0, max(pemf_w)], [0, max(unbc_w)], "k--", alpha=0.3)

    # Label pain AUs
    for i in PAIN_IDX:
        name = au_names[i] if i < len(au_names) else f"AU_{i}"
        ax2.annotate(name, (pemf_w[i], unbc_w[i]),
                     fontsize=8, fontweight="bold", color="#EF5350",
                     xytext=(5, 5), textcoords="offset points")

    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cross_dataset_transfer.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: cross_dataset_transfer.png")

    # Save
    with open(f"{RESULTS_DIR}/cross_dataset_results.txt", "w") as f:
        f.write("Cross-Dataset Transfer Results\n")
        f.write(f"BAUD-Linear (PEMF→PEMF): AUC={pemf_on_pemf['auc']:.4f}\n")
        f.write(f"BAUD-Linear (PEMF→UNBC): AUC={pemf_on_unbc['auc']:.4f}\n")
        f.write(f"BAUD-Linear (UNBC→UNBC): AUC={unbc_on_unbc['auc']:.4f}\n")
        f.write(f"BAUD-Linear (UNBC→PEMF): AUC={unbc_on_pemf['auc']:.4f}\n")
        f.write(f"Weight correlation: r={corr:.4f}\n")
    print(f"  Saved: cross_dataset_results.txt")

    print(f"\n{'='*70}")
    print(f"  ✅ CROSS-DATASET TRANSFER COMPLETE")
    print(f"{'='*70}")
    print(f"  📤 Share console output + plot!")


if __name__ == "__main__":
    main()
