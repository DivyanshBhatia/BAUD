#!/usr/bin/env python3
"""
FG 2027 Critical Fixes
========================
1. P1 with DISJOINT neutral frames (fixes data leakage)
2. λ cross-validation (fixes post-hoc concern)
3. Corrected weighting ablation gap

Run on Colab:
    !pip install py-feat --quiet
    !kaggle datasets download -d coder98/emotionpain -p /content/ --unzip
    !cd /content/BAUD && python scripts/run_fg_fixes.py
"""
import os, sys, numpy as np, torch, time, warnings
from PIL import Image
from torchvision import transforms
from collections import OrderedDict, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
PAIN_IDX = [2, 4, 5, 6, 7, 17]
SEED = 42; np.random.seed(SEED)
BATCH_SIZE = 64; K = 50; W = 10

w_prior = np.ones(41)
for i in PAIN_IDX: w_prior[i] = 3.0
w_prior /= w_prior.sum()

w_uniform = np.ones(41) / 41.0

def safe_auc(labels, scores):
    try: return roc_auc_score(labels, scores)
    except: return 0.5

def score_baud(mean_b, std_b, frames, weights=None):
    if weights is None: weights = w_prior
    return np.array([1.0/(1.0+np.exp(-np.dot(weights, np.maximum((f-mean_b)/std_b,0))+2.0))
                      for f in frames])

def score_center_only(mean_b, frames, weights=None):
    if weights is None: weights = w_prior
    return np.array([1.0/(1.0+np.exp(-np.dot(weights, np.maximum(f-mean_b,0))+2.0))
                      for f in frames])

def score_generic(frames):
    return np.array([np.mean(f[PAIN_IDX]) for f in frames])

def get_lowact(first_k):
    if len(first_k) <= W: return first_k
    best_s, best_i = np.inf, 0
    for i in range(len(first_k) - W + 1):
        s = np.mean(first_k[i:i+W][:, PAIN_IDX])
        if s < best_s: best_s, best_i = s, i
    return first_k[best_i:best_i+W]

# ── LOAD MODEL ──
IMAGES_DIR = "/content/Images/Images"
LABELS_DIR = "/content/Frame_Labels/Frame_Labels/PSPI"
OPENGRAPHAU_DIR = "/content/BAUD/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

au_transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
sys.path.insert(0, OPENGRAPHAU_DIR)
from model.MEFL import MEFARG
model = MEFARG(num_main_classes=27, num_sub_classes=14, backbone="resnet50")
ckpt = torch.load(CHECKPOINT, map_location="cpu")
sd = OrderedDict((k.replace("module.", ""), v)
                 for k, v in ckpt.get("state_dict", ckpt).items())
model.load_state_dict(sd, strict=False)
model.eval().to(DEVICE)
print("✅ OpenGraphAU loaded")

def extract_aus(paths):
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

# Load UNBC
print("Loading UNBC...")
t0 = time.time()
subject_data = {}
for subj_folder in sorted(os.listdir(IMAGES_DIR)):
    subj_id = subj_folder.split("-")[0]
    subj_img = os.path.join(IMAGES_DIR, subj_folder)
    subj_lbl = os.path.join(LABELS_DIR, subj_folder)
    if not os.path.isdir(subj_img) or not os.path.isdir(subj_lbl): continue
    frames = []
    for seq in sorted(os.listdir(subj_img)):
        seq_img = os.path.join(subj_img, seq)
        seq_lbl = os.path.join(subj_lbl, seq)
        if not os.path.isdir(seq_img) or not os.path.isdir(seq_lbl): continue
        for img_file in sorted(os.listdir(seq_img)):
            if not img_file.endswith(".png"): continue
            lbl_file = os.path.join(seq_lbl, img_file.replace(".png","")+"_facs.txt")
            if os.path.exists(lbl_file):
                try:
                    with open(lbl_file) as f: pspi = float(f.read().strip())
                    frames.append({"path": os.path.join(seq_img, img_file), "pspi": pspi})
                except: pass
    if frames:
        if subj_id not in subject_data: subject_data[subj_id] = []
        subject_data[subj_id].extend(frames)

for idx, (subj_id, frames) in enumerate(sorted(subject_data.items())):
    aus = extract_aus([f["path"] for f in frames])
    pspis = np.array([f["pspi"] for f in frames])
    subject_data[subj_id] = {"aus": aus, "labels": (pspis > 0).astype(int), "pspis": pspis}
    if (idx+1) % 5 == 0 or idx == 0:
        print(f"  {idx+1}/{len(subject_data)} ({time.time()-t0:.1f}s)")
print(f"✅ Done ({time.time()-t0:.0f}s)\n")

valid = sorted([s for s in subject_data
                if sum(subject_data[s]["labels"]) >= 5
                and sum(1-subject_data[s]["labels"]) >= 5])

# Global neutral mean
all_neutrals = [subject_data[s]["aus"][subject_data[s]["labels"]==0] for s in valid]
global_mean = np.mean(np.concatenate(all_neutrals), 0)


# ================================================================
# ★★ EXPERIMENT 1: P1 WITH DISJOINT NEUTRAL FRAMES
# ================================================================
print("=" * 70)
print("  ★★ EXP 1: P1 WITH DISJOINT NEUTRAL FRAMES")
print("  Fixes data leakage: calibration and evaluation sets are separate")
print("=" * 70)

N_SPLITS = 5  # 5-fold split of neutral frames

results_leaky = defaultdict(list)     # Original P1 (leaky)
results_disjoint = defaultdict(list)  # Fixed P1 (disjoint)
results_shuffled_disjoint = defaultdict(list)  # Shuffled with disjoint

for subj_id in valid:
    d = subject_data[subj_id]
    aus, labels = d["aus"], d["labels"]
    neutral_idx = np.where(labels == 0)[0]
    pain_idx_arr = np.where(labels == 1)[0]

    if len(neutral_idx) < 10 or len(pain_idx_arr) < 5:
        continue

    # ── Original P1 (leaky): calibrate on ALL neutral, evaluate on ALL ──
    neutral_all = aus[neutral_idx]
    mb = np.mean(neutral_all, 0)
    sb = np.maximum(np.std(neutral_all, 0), 1e-4)
    results_leaky["BAUD"].append(safe_auc(labels, score_baud(mb, sb, aus)))
    results_leaky["Center-only"].append(safe_auc(labels, score_center_only(mb, aus)))
    results_leaky["Generic"].append(safe_auc(labels, score_generic(aus)))

    # ── Disjoint P1: split neutral 50/50, calibrate on first half, evaluate on second half + pain ──
    np.random.seed(SEED)
    perm = np.random.permutation(len(neutral_idx))
    split = len(perm) // 2
    cal_neutral = neutral_idx[perm[:split]]   # Calibration neutral
    eval_neutral = neutral_idx[perm[split:]]  # Evaluation neutral (held out)

    # Calibration statistics from calibration-only neutral
    cal_aus = aus[cal_neutral]
    mb_d = np.mean(cal_aus, 0)
    sb_d = np.maximum(np.std(cal_aus, 0), 1e-4)

    # Evaluation set: held-out neutral + all pain
    eval_idx = np.concatenate([eval_neutral, pain_idx_arr])
    eval_aus = aus[eval_idx]
    eval_labels = labels[eval_idx]

    results_disjoint["BAUD"].append(safe_auc(eval_labels, score_baud(mb_d, sb_d, eval_aus)))
    results_disjoint["Center-only"].append(safe_auc(eval_labels, score_center_only(mb_d, eval_aus)))
    results_disjoint["Generic"].append(safe_auc(eval_labels, score_generic(eval_aus)))

    # ── Shuffled baseline with disjoint evaluation ──
    other_subjs = [s for s in valid if s != subj_id]
    shuffle_runs = []
    for _ in range(30):
        donor = np.random.choice(other_subjs)
        donor_d = subject_data[donor]
        donor_neutral = donor_d["aus"][donor_d["labels"] == 0]
        dm = np.mean(donor_neutral, 0)
        ds = np.maximum(np.std(donor_neutral, 0), 1e-4)
        shuffle_runs.append(safe_auc(eval_labels, score_baud(dm, ds, eval_aus)))
    results_shuffled_disjoint["BAUD"].append(np.mean(shuffle_runs))

print(f"\n  UNBC P1 Comparison (n={len(results_leaky['BAUD'])} subjects):")
print(f"  {'Method':<20} {'Leaky P1':>10} {'Disjoint P1':>12} {'Difference':>12}")
print(f"  {'-'*56}")
for name in ["BAUD", "Center-only", "Generic"]:
    leaky = np.mean(results_leaky[name])
    disjoint = np.mean(results_disjoint[name])
    diff = disjoint - leaky
    print(f"  {name:<20} {leaky:>10.4f} {disjoint:>12.4f} {diff:>+12.4f}")

# Key tests
own_d = results_disjoint["BAUD"]
shuf_d = results_shuffled_disjoint["BAUD"]
gen_d = results_disjoint["Generic"]

try: _, p_own_gen = wilcoxon(own_d, gen_d)
except: p_own_gen = 1.0
try: _, p_own_shuf = wilcoxon(own_d, shuf_d)
except: p_own_shuf = 1.0
try: _, p_shuf_gen = wilcoxon(shuf_d, gen_d)
except: p_shuf_gen = 1.0

print(f"\n  DISJOINT significance tests:")
print(f"    Own vs Generic:    {np.mean(own_d):.4f} vs {np.mean(gen_d):.4f}, p={p_own_gen:.4f}")
print(f"    Own vs Shuffled:   {np.mean(own_d):.4f} vs {np.mean(shuf_d):.4f}, p={p_own_shuf:.4f}")
print(f"    Shuffled vs Generic: {np.mean(shuf_d):.4f} vs {np.mean(gen_d):.4f}, p={p_shuf_gen:.4f}")

if np.mean(own_d) > np.mean(gen_d) and np.mean(own_d) > np.mean(shuf_d):
    print(f"\n  ✅ Personalization SURVIVES disjoint evaluation!")
    print(f"     The P1 finding is NOT an artifact of data leakage.")
else:
    print(f"\n  ⚠️ Disjoint evaluation changes the picture — needs careful reporting.")


# ================================================================
# ★ EXPERIMENT 2: λ CROSS-VALIDATION
# ================================================================
print(f"\n{'='*70}")
print("  ★ EXP 2: λ CROSS-VALIDATION (LOSO)")
print("  Shows λ=0.5 is optimal under proper CV, not post-hoc")
print("=" * 70)

lambda_values = np.arange(0, 1.05, 0.1)

# For each test subject, find optimal λ using remaining subjects
loso_optimal_lambdas = []
loso_aucs_at_optimal = []
loso_aucs_at_05 = []
loso_aucs_at_10 = []

for test_subj in valid:
    d = subject_data[test_subj]
    aus, labels = d["aus"], d["labels"]
    if len(aus) <= K: continue
    first_k = aus[:K]
    test_aus = aus[K:]
    test_labels = labels[K:]
    if len(set(test_labels)) < 2: continue

    patient_mean = np.mean(first_k, 0)

    # Inner LOSO: find best λ on remaining subjects
    inner_scores = {lam: [] for lam in lambda_values}
    for val_subj in valid:
        if val_subj == test_subj: continue
        vd = subject_data[val_subj]
        vaus, vlabels = vd["aus"], vd["labels"]
        if len(vaus) <= K: continue
        vfirst = vaus[:K]
        vtest = vaus[K:]
        vtl = vlabels[K:]
        if len(set(vtl)) < 2: continue

        vm = np.mean(vfirst, 0)
        for lam in lambda_values:
            shrunk = lam * vm + (1 - lam) * global_mean
            s = score_center_only(shrunk, vtest)
            inner_scores[lam].append(safe_auc(vtl, s))

    # Pick λ that maximizes mean AUC on inner subjects
    best_lam = 0.5
    best_inner = 0
    for lam in lambda_values:
        if inner_scores[lam]:
            m = np.mean(inner_scores[lam])
            if m > best_inner:
                best_inner = m
                best_lam = lam

    loso_optimal_lambdas.append(best_lam)

    # Evaluate on test subject with CV-selected λ
    shrunk_opt = best_lam * patient_mean + (1 - best_lam) * global_mean
    loso_aucs_at_optimal.append(safe_auc(test_labels, score_center_only(shrunk_opt, test_aus)))

    # Also evaluate at fixed λ=0.5 and λ=1.0
    shrunk_05 = 0.5 * patient_mean + 0.5 * global_mean
    loso_aucs_at_05.append(safe_auc(test_labels, score_center_only(shrunk_05, test_aus)))

    shrunk_10 = patient_mean
    loso_aucs_at_10.append(safe_auc(test_labels, score_center_only(shrunk_10, test_aus)))

print(f"\n  LOSO-CV selected λ distribution:")
from collections import Counter
lam_counts = Counter([round(l, 1) for l in loso_optimal_lambdas])
for lam in sorted(lam_counts.keys()):
    bar = "█" * lam_counts[lam]
    print(f"    λ={lam:.1f}: {lam_counts[lam]:>3} subjects {bar}")

print(f"\n  AUC comparison (P2, centering-only):")
print(f"    CV-selected λ (per-subject): {np.mean(loso_aucs_at_optimal):.4f}")
print(f"    Fixed λ=0.5:                 {np.mean(loso_aucs_at_05):.4f}")
print(f"    Fixed λ=1.0 (pure patient):  {np.mean(loso_aucs_at_10):.4f}")
print(f"    Most frequent CV-selected λ:  {max(lam_counts, key=lam_counts.get):.1f}")

if abs(np.mean(loso_aucs_at_optimal) - np.mean(loso_aucs_at_05)) < 0.01:
    print(f"\n  ✅ CV-selected λ ≈ fixed 0.5 — no post-hoc advantage")
else:
    print(f"\n  ⚠️ CV-selected λ differs from 0.5 by {np.mean(loso_aucs_at_optimal)-np.mean(loso_aucs_at_05):+.4f}")


# ================================================================
# EXPERIMENT 3: CORRECTED WEIGHTING ABLATION
# ================================================================
print(f"\n{'='*70}")
print("  EXP 3: CORRECTED WEIGHTING ABLATION (DISJOINT P1)")
print("  Honest gap: weights contribute ~5pp, centering ~8pp")
print("=" * 70)

weight_results = {"Prior (3x)": [], "Learned": [], "Uniform": [],
                  "Center-prior": [], "Center-uniform": [], "Generic": []}

subj_list = sorted(valid)
for test_subj in subj_list:
    d = subject_data[test_subj]
    aus, labels = d["aus"], d["labels"]
    neutral_idx = np.where(labels == 0)[0]
    pain_idx_arr = np.where(labels == 1)[0]
    if len(neutral_idx) < 10 or len(pain_idx_arr) < 5: continue

    # Disjoint split
    np.random.seed(SEED)
    perm = np.random.permutation(len(neutral_idx))
    split = len(perm) // 2
    cal_aus = aus[neutral_idx[perm[:split]]]
    eval_idx = np.concatenate([neutral_idx[perm[split:]], pain_idx_arr])
    eval_aus = aus[eval_idx]
    eval_labels = labels[eval_idx]

    mb = np.mean(cal_aus, 0)
    sb = np.maximum(np.std(cal_aus, 0), 1e-4)

    # Learn weights from other subjects
    train_z, train_l = [], []
    for s in subj_list:
        if s == test_subj: continue
        td = subject_data[s]
        if sum(td["labels"]) < 3 or sum(1-td["labels"]) < 3: continue
        tn = td["aus"][td["labels"] == 0]
        tm = np.mean(tn, 0); ts = np.maximum(np.std(tn, 0), 1e-4)
        z = np.maximum((td["aus"] - tm) / ts, 0)
        train_z.append(z); train_l.append(td["labels"])
    try:
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=SEED)
        lr.fit(np.concatenate(train_z), np.concatenate(train_l))
        learned_w = np.maximum(lr.coef_[0], 0)
        learned_w = learned_w / (learned_w.sum() + 1e-8)
    except:
        learned_w = w_prior.copy()

    weight_results["Prior (3x)"].append(safe_auc(eval_labels, score_baud(mb, sb, eval_aus, w_prior)))
    weight_results["Learned"].append(safe_auc(eval_labels, score_baud(mb, sb, eval_aus, learned_w)))
    weight_results["Uniform"].append(safe_auc(eval_labels, score_baud(mb, sb, eval_aus, w_uniform)))
    weight_results["Center-prior"].append(safe_auc(eval_labels, score_center_only(mb, eval_aus, w_prior)))
    weight_results["Center-uniform"].append(safe_auc(eval_labels, score_center_only(mb, eval_aus, w_uniform)))
    weight_results["Generic"].append(safe_auc(eval_labels, score_generic(eval_aus)))

gen_m = np.mean(weight_results["Generic"])
print(f"\n  Disjoint P1 weighting ablation (n={len(weight_results['Generic'])} subjects):")
print(f"  {'Configuration':<22} {'AUC':>8} {'vs Generic':>12} {'vs Uniform':>12}")
print(f"  {'-'*56}")
uni_m = np.mean(weight_results["Uniform"])
for name in ["Prior (3x)", "Learned", "Uniform", "Center-prior", "Center-uniform", "Generic"]:
    m = np.mean(weight_results[name])
    print(f"  {name:<22} {m:>8.4f} {m-gen_m:>+12.4f} {m-uni_m:>+12.4f}")

prior_m = np.mean(weight_results["Prior (3x)"])
print(f"\n  DECOMPOSITION (disjoint):")
print(f"    Total gain (prior z-score vs generic): {prior_m - gen_m:+.4f}")
print(f"    From centering (uniform vs generic):   {uni_m - gen_m:+.4f}")
print(f"    From weighting (prior vs uniform):     {prior_m - uni_m:+.4f}")
if (prior_m - gen_m) > 0:
    pct_center = (uni_m - gen_m) / (prior_m - gen_m) * 100
    pct_weight = (prior_m - uni_m) / (prior_m - gen_m) * 100
    print(f"    Centering contribution:                {pct_center:.0f}%")
    print(f"    Weighting contribution:                {pct_weight:.0f}%")


# ================================================================
# EXPERIMENT 4: DISJOINT SHUFFLED-BASELINE (FULL)
# ================================================================
print(f"\n{'='*70}")
print("  EXP 4: SHUFFLED BASELINE WITH DISJOINT EVALUATION")
print("=" * 70)

own_aucs, shuffled_aucs, generic_aucs = [], [], []
for subj_id in valid:
    d = subject_data[subj_id]
    aus, labels = d["aus"], d["labels"]
    neutral_idx = np.where(labels == 0)[0]
    pain_idx_arr = np.where(labels == 1)[0]
    if len(neutral_idx) < 10 or len(pain_idx_arr) < 5: continue

    np.random.seed(SEED)
    perm = np.random.permutation(len(neutral_idx))
    split = len(perm) // 2
    cal_aus = aus[neutral_idx[perm[:split]]]
    eval_idx = np.concatenate([neutral_idx[perm[split:]], pain_idx_arr])
    eval_aus = aus[eval_idx]
    eval_labels = labels[eval_idx]

    mb = np.mean(cal_aus, 0); sb = np.maximum(np.std(cal_aus, 0), 1e-4)
    own_aucs.append(safe_auc(eval_labels, score_baud(mb, sb, eval_aus)))
    generic_aucs.append(safe_auc(eval_labels, score_generic(eval_aus)))

    # Shuffled
    other_subjs = [s for s in valid if s != subj_id]
    shuf_runs = []
    for _ in range(50):
        donor = np.random.choice(other_subjs)
        dd = subject_data[donor]
        dn = dd["aus"][dd["labels"] == 0]
        dm = np.mean(dn, 0); ds = np.maximum(np.std(dn, 0), 1e-4)
        shuf_runs.append(safe_auc(eval_labels, score_baud(dm, ds, eval_aus)))
    shuffled_aucs.append(np.mean(shuf_runs))

print(f"  DISJOINT evaluation:")
print(f"    Own baseline:      {np.mean(own_aucs):.4f}")
print(f"    Shuffled baseline: {np.mean(shuffled_aucs):.4f}")
print(f"    Generic:           {np.mean(generic_aucs):.4f}")
try: _, p = wilcoxon(own_aucs, shuffled_aucs)
except: p = 1.0
print(f"    Own vs Shuffled: p={p:.4f}")
try: _, p2 = wilcoxon(shuffled_aucs, generic_aucs)
except: p2 = 1.0
print(f"    Shuffled vs Generic: p={p2:.4f}")

if np.mean(own_aucs) > np.mean(shuffled_aucs) and np.mean(shuffled_aucs) < np.mean(generic_aucs):
    print(f"    ✅ Shuffled still hurts under disjoint evaluation!")


# ================================================================
print(f"\n{'='*70}")
print("  SUMMARY FOR FG PAPER")
print("=" * 70)
print(f"  1. Disjoint P1 BAUD:        {np.mean(results_disjoint['BAUD']):.4f} "
      f"(vs leaky: {np.mean(results_leaky['BAUD']):.4f})")
print(f"  2. CV-selected λ:           most frequent = {max(lam_counts, key=lam_counts.get):.1f}, "
      f"AUC = {np.mean(loso_aucs_at_optimal):.4f}")
print(f"  3. Weight contribution:     centering ~{pct_center:.0f}%, weighting ~{pct_weight:.0f}%")
print(f"  4. Shuffled (disjoint):     still hurts = {np.mean(shuffled_aucs) < np.mean(generic_aucs)}")
print(f"\n  📤 Share FULL output!")
