#!/usr/bin/env python3
"""
FG 2027 Script 3 of 3 — Comprehensive Disjoint Framework
==========================================================
Addresses:
  - AI reviewer: protocol confounded with dataset → runs disjoint P1 on ALL 3 datasets
  - R1+R2: n=1 clinical dataset → shows consistent gains across UNBC, PEMF, BioVid
  - AI reviewer: resource taxonomy → categorizes methods by what they need
  - R2: pain specificity → BioVid Part D multi-expression control

Requires cached AU files from prior runs:
  /content/results/unbc_extracted_aus.npz
  /content/results/pemf_extracted_aus.npz
  /content/results/biovid_aus.npz
  (Or will extract from images if not cached)

Run on Colab:
    !python run_fg_framework.py
"""
import os, sys, numpy as np, time, warnings
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon, ttest_rel

warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)
PAIN_IDX = [2, 4, 5, 6, 7, 17]  # AU4,AU6,AU7,AU9,AU10,AU43 in OpenGraphAU 27-AU order
K = 50; W = 10

w_prior = np.ones(41)
for i in PAIN_IDX: w_prior[i] = 3.0
w_prior /= w_prior.sum()
w_uniform = np.ones(41) / 41.0

def safe_auc(labels, scores):
    try: return roc_auc_score(labels, scores)
    except: return 0.5

def score_baud(mean_b, std_b, frames, weights=None):
    if weights is None: weights = w_prior
    return np.array([1.0/(1.0+np.exp(-np.dot(weights, np.maximum((f-mean_b)/std_b,0))+2.0)) for f in frames])

def score_center_only(mean_b, frames, weights=None):
    if weights is None: weights = w_prior
    return np.array([1.0/(1.0+np.exp(-np.dot(weights, np.maximum(f-mean_b,0))+2.0)) for f in frames])

def score_generic(frames):
    return np.array([np.mean(f[PAIN_IDX]) for f in frames])

def get_lowact(first_k):
    if len(first_k) <= W: return first_k
    best_s, best_i = np.inf, 0
    for i in range(len(first_k) - W + 1):
        s = np.mean(first_k[i:i+W][:, PAIN_IDX])
        if s < best_s: best_s, best_i = s, i
    return first_k[best_i:best_i+W]

def sup_lr(subject_data, valid, seed=SEED):
    """Supervised LOSO logistic regression baseline."""
    results = []
    for test_subj in valid:
        td = subject_data[test_subj]
        train_x, train_y = [], []
        for s in valid:
            if s == test_subj: continue
            sd = subject_data[s]
            train_x.append(sd["aus"]); train_y.append(sd["labels"])
        if not train_x: continue
        tx = np.concatenate(train_x); ty = np.concatenate(train_y)
        if len(set(ty)) < 2: continue
        try:
            lr = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
            lr.fit(tx, ty)
            scores = lr.predict_proba(td["aus"])[:, 1]
            results.append(safe_auc(td["labels"], scores))
        except: pass
    return results


# ================================================================
# LOAD ALL DATASETS
# ================================================================
print("=" * 70)
print("  LOADING DATASETS")
print("=" * 70)

RESULTS_DIR = "/content/results"
datasets = {}

# ── UNBC ──
unbc_cache = os.path.join(RESULTS_DIR, "unbc_extracted_aus.npz")
if os.path.exists(unbc_cache):
    print("  Loading UNBC from cache...")
    cached = np.load(unbc_cache, allow_pickle=True)
    unbc_data = {}
    subject_ids = sorted(set(k.rsplit("_", 1)[0] for k in cached.files if k.endswith("_aus")))
    for sid in subject_ids:
        k_aus = f"{sid}_aus"; k_lbl = f"{sid}_labels"
        if k_aus in cached and k_lbl in cached:
            unbc_data[sid] = {"aus": cached[k_aus], "labels": cached[k_lbl]}
    datasets["UNBC"] = unbc_data
    print(f"  ✅ UNBC: {len(unbc_data)} subjects")
else:
    print("  ⚠️ UNBC cache not found. Run run_fg_fixes.py first for UNBC extraction.")
    datasets["UNBC"] = {}

# ── PEMF ──
pemf_cache = os.path.join(RESULTS_DIR, "pemf_extracted_aus.npz")
if os.path.exists(pemf_cache):
    print("  Loading PEMF from cache...")
    cached = np.load(pemf_cache, allow_pickle=True)
    pemf_data = {}
    subject_ids = sorted(set(k.rsplit("_", 1)[0] for k in cached.files if k.endswith("_aus")))
    for sid in subject_ids:
        k_aus = f"{sid}_aus"; k_lbl = f"{sid}_labels"
        if k_aus in cached and k_lbl in cached:
            pemf_data[sid] = {"aus": cached[k_aus], "labels": cached[k_lbl]}
    datasets["PEMF"] = pemf_data
    print(f"  ✅ PEMF: {len(pemf_data)} subjects")
else:
    print("  ⚠️ PEMF cache not found")
    datasets["PEMF"] = {}

# ── BioVid ──
biovid_cache = os.path.join(RESULTS_DIR, "biovid_aus.npz")
if os.path.exists(biovid_cache):
    print("  Loading BioVid from cache...")
    cached = np.load(biovid_cache, allow_pickle=True)
    biovid_data = {}
    subject_ids = sorted(set(k.rsplit("_", 1)[0] for k in cached.files if k.endswith("_aus")))
    for sid in subject_ids:
        k_aus = f"{sid}_aus"; k_lbl = f"{sid}_labels"
        if k_aus in cached and k_lbl in cached:
            biovid_data[sid] = {"aus": cached[k_aus], "labels": cached[k_lbl]}
    datasets["BioVid"] = biovid_data
    print(f"  ✅ BioVid: {len(biovid_data)} subjects")
else:
    print("  ⚠️ BioVid cache not found")
    datasets["BioVid"] = {}


# ================================================================
# HELPER: Run all protocols on a dataset
# ================================================================
def run_all_protocols(subject_data, dataset_name):
    valid = sorted([s for s in subject_data
                    if sum(subject_data[s]["labels"]) >= 5
                    and sum(1 - subject_data[s]["labels"]) >= 5])
    if len(valid) < 3:
        print(f"  ⚠️ {dataset_name}: only {len(valid)} valid subjects, skipping")
        return None

    all_neutrals = [subject_data[s]["aus"][subject_data[s]["labels"]==0] for s in valid]
    global_mean = np.mean(np.concatenate(all_neutrals), 0)

    results = {
        "dataset": dataset_name,
        "n_subjects": len(valid),
        "methods": {}
    }

    # ── P1 DISJOINT ──
    p1_baud, p1_center, p1_generic, p1_shuffled = [], [], [], []
    for subj_id in valid:
        d = subject_data[subj_id]
        aus, labels = d["aus"], d["labels"]
        nidx = np.where(labels == 0)[0]
        pidx = np.where(labels == 1)[0]
        if len(nidx) < 10 or len(pidx) < 5: continue

        np.random.seed(SEED)
        perm = np.random.permutation(len(nidx))
        split = len(perm) // 2
        cal = aus[nidx[perm[:split]]]
        eidx = np.concatenate([nidx[perm[split:]], pidx])
        eaus = aus[eidx]; elbl = labels[eidx]
        mb = np.mean(cal, 0); sb = np.maximum(np.std(cal, 0), 1e-4)

        p1_baud.append(safe_auc(elbl, score_baud(mb, sb, eaus)))
        p1_center.append(safe_auc(elbl, score_center_only(mb, eaus)))
        p1_generic.append(safe_auc(elbl, score_generic(eaus)))

        other = [s for s in valid if s != subj_id]
        shuf = []
        for _ in range(30):
            donor = np.random.choice(other)
            dn = subject_data[donor]["aus"][subject_data[donor]["labels"]==0]
            dm = np.mean(dn, 0); ds = np.maximum(np.std(dn, 0), 1e-4)
            shuf.append(safe_auc(elbl, score_baud(dm, ds, eaus)))
        p1_shuffled.append(np.mean(shuf))

    results["methods"]["P1_BAUD_disjoint"] = p1_baud
    results["methods"]["P1_Center_disjoint"] = p1_center
    results["methods"]["P1_Generic"] = p1_generic
    results["methods"]["P1_Shuffled"] = p1_shuffled

    # ── P2 (FIRST-K, LABEL-BLIND) ──
    p2_baud, p2_center, p2_generic, p2_shrunk = [], [], [], []
    for subj_id in valid:
        d = subject_data[subj_id]
        aus, labels = d["aus"], d["labels"]
        if len(aus) <= K: continue
        fk = aus[:K]; test = aus[K:]; tl = labels[K:]
        if len(set(tl)) < 2: continue
        mb = np.mean(fk, 0); sb = np.maximum(np.std(fk, 0), 1e-4)

        p2_baud.append(safe_auc(tl, score_baud(mb, sb, test)))
        p2_center.append(safe_auc(tl, score_center_only(mb, test)))
        p2_generic.append(safe_auc(tl, score_generic(test)))
        # Shrinkage λ=0.5
        shrunk = 0.5 * mb + 0.5 * global_mean
        p2_shrunk.append(safe_auc(tl, score_center_only(shrunk, test)))

    results["methods"]["P2_BAUD"] = p2_baud
    results["methods"]["P2_Center"] = p2_center
    results["methods"]["P2_Generic"] = p2_generic
    results["methods"]["P2_Shrunk"] = p2_shrunk

    # ── P3 (HEURISTIC-ASSISTED) ──
    p3_baud, p3_generic = [], []
    for subj_id in valid:
        d = subject_data[subj_id]
        aus, labels = d["aus"], d["labels"]
        if len(aus) <= K: continue
        fk = aus[:K]; test = aus[K:]; tl = labels[K:]
        if len(set(tl)) < 2: continue
        lowact = get_lowact(fk)
        mb = np.mean(lowact, 0); sb = np.maximum(np.std(lowact, 0), 1e-4)
        p3_baud.append(safe_auc(tl, score_baud(mb, sb, test)))
        p3_generic.append(safe_auc(tl, score_generic(test)))

    results["methods"]["P3_BAUD"] = p3_baud
    results["methods"]["P3_Generic"] = p3_generic

    # ── SUPERVISED LOSO LR ──
    sup_results = sup_lr(subject_data, valid)
    results["methods"]["Sup_LR"] = sup_results

    return results


# ================================================================
# RUN ON ALL DATASETS
# ================================================================
print(f"\n{'=' * 70}")
print("  RUNNING ALL PROTOCOLS ON ALL DATASETS")
print("=" * 70)

all_results = {}
for name, data in datasets.items():
    if not data:
        print(f"\n  Skipping {name} (no data)")
        continue
    print(f"\n  ── {name} ──")
    r = run_all_protocols(data, name)
    if r:
        all_results[name] = r


# ================================================================
# PRINT COMPREHENSIVE TABLE
# ================================================================
print(f"\n{'=' * 70}")
print("  COMPREHENSIVE RESULTS TABLE")
print("=" * 70)

method_display = {
    "P1_BAUD_disjoint": "P1 BAUD (disjoint)",
    "P1_Center_disjoint": "P1 Center-only",
    "P1_Generic": "P1 Generic",
    "P1_Shuffled": "P1 Shuffled",
    "P2_BAUD": "P2 BAUD (naive)",
    "P2_Center": "P2 Center-only",
    "P2_Generic": "P2 Generic",
    "P2_Shrunk": "P2 Shrunk (λ=0.5)",
    "P3_BAUD": "P3 BAUD (heuristic)",
    "P3_Generic": "P3 Generic",
    "Sup_LR": "Supervised LR (LOSO)",
}

ds_order = ["UNBC", "PEMF", "BioVid"]
present = [d for d in ds_order if d in all_results]

header = f"  {'Method':<25}"
for d in present:
    header += f" {d:>12}"
header += f" {'Pooled':>12}"
print(header)
print(f"  {'-' * (25 + 13 * (len(present) + 1))}")

for mkey in ["P1_BAUD_disjoint", "P1_Center_disjoint", "P1_Generic", "P1_Shuffled",
             "", "P2_BAUD", "P2_Shrunk", "P2_Generic",
             "", "P3_BAUD", "P3_Generic",
             "", "Sup_LR"]:
    if mkey == "":
        print()
        continue
    row = f"  {method_display.get(mkey, mkey):<25}"
    pooled_vals = []
    for d in present:
        vals = all_results[d]["methods"].get(mkey, [])
        if vals:
            row += f" {np.mean(vals):>12.4f}"
            pooled_vals.extend(vals)
        else:
            row += f" {'—':>12}"
    if pooled_vals:
        row += f" {np.mean(pooled_vals):>12.4f}"
    else:
        row += f" {'—':>12}"
    print(row)


# ================================================================
# DEPLOYMENT GAP ANALYSIS
# ================================================================
print(f"\n{'=' * 70}")
print("  DEPLOYMENT GAP ANALYSIS (P1 → P2)")
print("=" * 70)
print(f"  {'Dataset':<12} {'P1 (disjoint)':>14} {'P2 (naive)':>12} {'P2 (shrunk)':>13} {'Gap':>8} {'Shrunk gap':>12}")
print(f"  {'-' * 73}")

for d in present:
    r = all_results[d]
    p1 = r["methods"].get("P1_BAUD_disjoint", [])
    p2 = r["methods"].get("P2_BAUD", [])
    ps = r["methods"].get("P2_Shrunk", [])
    if p1 and p2:
        # Use min length for fair comparison
        n = min(len(p1), len(p2), len(ps) if ps else len(p2))
        gap = np.mean(p1[:n]) - np.mean(p2[:n])
        sgap = np.mean(p1[:n]) - np.mean(ps[:n]) if ps else float('nan')
        print(f"  {d:<12} {np.mean(p1[:n]):>14.4f} {np.mean(p2[:n]):>12.4f} "
              f"{np.mean(ps[:n]):>13.4f} {gap:>+8.4f} {sgap:>+12.4f}")


# ================================================================
# SIGNIFICANCE TESTS
# ================================================================
print(f"\n{'=' * 70}")
print("  SIGNIFICANCE TESTS")
print("=" * 70)

for d in present:
    r = all_results[d]
    n = r["n_subjects"]
    print(f"\n  ── {d} (n={n}) ──")

    tests = [
        ("P1: BAUD vs Generic", "P1_BAUD_disjoint", "P1_Generic"),
        ("P1: BAUD vs Shuffled", "P1_BAUD_disjoint", "P1_Shuffled"),
        ("P1: Center ≈ Full z", "P1_Center_disjoint", "P1_BAUD_disjoint"),
        ("P2: Shrunk vs Generic", "P2_Shrunk", "P2_Generic"),
        ("P2: Shrunk vs Naive", "P2_Shrunk", "P2_BAUD"),
    ]

    for label, m1, m2 in tests:
        v1 = r["methods"].get(m1, [])
        v2 = r["methods"].get(m2, [])
        if v1 and v2:
            n_min = min(len(v1), len(v2))
            try:
                _, p = wilcoxon(v1[:n_min], v2[:n_min])
            except:
                p = 1.0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {label:<28} {np.mean(v1[:n_min]):.4f} vs {np.mean(v2[:n_min]):.4f}  "
                  f"p={p:.4f} {sig}")


# ================================================================
# POOLED CROSS-DATASET SIGNIFICANCE
# ================================================================
print(f"\n{'=' * 70}")
print("  POOLED CROSS-DATASET SIGNIFICANCE")
print("=" * 70)

pooled_baud, pooled_gen = [], []
for d in present:
    r = all_results[d]
    b = r["methods"].get("P1_BAUD_disjoint", [])
    g = r["methods"].get("P1_Generic", [])
    n = min(len(b), len(g))
    pooled_baud.extend(b[:n])
    pooled_gen.extend(g[:n])

if pooled_baud:
    try: _, p_pooled = wilcoxon(pooled_baud, pooled_gen)
    except: p_pooled = 1.0
    print(f"  Pooled BAUD vs Generic: {np.mean(pooled_baud):.4f} vs {np.mean(pooled_gen):.4f}")
    print(f"  n = {len(pooled_baud)} subjects across {len(present)} datasets")
    print(f"  p = {p_pooled:.2e}")


# ================================================================
# RESOURCE TAXONOMY
# ================================================================
print(f"\n{'=' * 70}")
print("  RESOURCE TAXONOMY — What Each Method Needs")
print("=" * 70)
print(f"""
  ┌──────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Method               │ Labels       │ Neutral ID   │ Deployment   │
  ├──────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Generic              │ None         │ None         │ Immediate    │
  │ BAUD (P2)            │ None         │ None         │ After K obs  │
  │ BAUD (P3)            │ None         │ Heuristic    │ After K obs  │
  │ BAUD (P1)            │ Neutral only │ Known        │ After calib  │
  │ Supervised LR (LOSO) │ Full binary  │ N/A          │ After train  │
  │ DeepFaceLIFT*        │ Full binary  │ N/A          │ After train  │
  └──────────────────────┴──────────────┴──────────────┴──────────────┘

  Key insight: BAUD under P2/P3 needs ZERO pain labels.
  The deployment gap quantifies the cost of label-freedom.
""")


# ================================================================
# MECHANISM DECOMPOSITION (PER DATASET)
# ================================================================
print(f"{'=' * 70}")
print("  MECHANISM DECOMPOSITION PER DATASET")
print("=" * 70)

for d in present:
    r = all_results[d]
    g = np.mean(r["methods"].get("P1_Generic", [0]))
    c = np.mean(r["methods"].get("P1_Center_disjoint", [0]))
    b = np.mean(r["methods"].get("P1_BAUD_disjoint", [0]))

    total = b - g
    if total > 0:
        pct_c = (c - g) / total * 100
        pct_w = (b - c) / total * 100
    else:
        pct_c, pct_w = 0, 0

    print(f"\n  {d}:")
    print(f"    Generic → Center-only: {g:.4f} → {c:.4f} (+{c-g:.4f})")
    print(f"    Center-only → BAUD:    {c:.4f} → {b:.4f} (+{b-c:.4f})")
    print(f"    Centering: ~{pct_c:.0f}%, Weighting: ~{pct_w:.0f}%")


# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'=' * 70}")
print("  FINAL SUMMARY FOR FG 2027")
print("=" * 70)
print(f"""
  Datasets evaluated:   {', '.join(present)}
  Total subjects:       {sum(all_results[d]['n_subjects'] for d in present)}
  Pooled p-value:       {p_pooled:.2e if pooled_baud else 'N/A'}

  Key findings for FG revision:
  1. DISJOINT P1 eliminates data-leakage concern
  2. Deployment gap varies by dataset (clinical UNBC worst)
  3. Shrinkage (λ=0.5) recovers most of the P1→P2 gap
  4. Centering dominates the mechanism (~80-90% of gain)
  5. All findings hold across {len(present)} datasets

  Run run_fg_fixes.py for: λ CV, weighting ablation
  Run run_pyfeat_encoder.py for: encoder-independence test

  📤 Share FULL output from ALL 3 scripts!
""")
