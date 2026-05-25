#!/usr/bin/env python3
"""
BAUD × PEMF × OpenGraphAU: Extract AUs from face images and run BAUD.
Fixed version with correct model parameters (27 main, 14 sub AUs).
"""
import os, sys, glob, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEMF_ROOT = "/content/pemf"
PICTURES_DIR = os.path.join(PEMF_ROOT, "pictures", "Pictures", "Modified")
OPENGRAPHAU_DIR = "/content/baud/external/OpenGraphAU"
CHECKPOINT = os.path.join(OPENGRAPHAU_DIR, "checkpoints",
                           "OpenGprahAU-ResNet50_second_stage.pth")
RESULTS_DIR = "/content/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_AUS = 41  # Model outputs 27 main + 14 sub = 41 total
PAIN_IDX = [2, 4, 5, 6, 7, 17]
PAIN_NAMES = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]
BATCH_SIZE = 32
EXPR_TYPES = ["Neutral", "Algometer Pain", "Laser Pain", "Posed Pain"]

au_transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model():
    sys.path.insert(0, OPENGRAPHAU_DIR)
    from model.MEFL import MEFARG
    # CORRECT: 27 main + 14 sub (matches checkpoint)
    model = MEFARG(num_main_classes=27, num_sub_classes=14, backbone="resnet50")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=False)
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    print(f"  ✅ OpenGraphAU loaded on {dev}")
    return model, dev

def extract_batch(model, dev, paths):
    tensors = []
    for p in paths:
        try:
            tensors.append(au_transform(Image.open(p).convert("RGB")))
        except:
            tensors.append(torch.zeros(3, 224, 224))
    batch = torch.stack(tensors).to(dev)
    with torch.no_grad():
        out = model(batch)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return torch.sigmoid(pred).cpu().numpy()

def extract_all(model, dev):
    print("\n🔍 Extracting AUs from PEMF images...")
    t0 = time.time()
    data = {}
    subjects = sorted([d for d in os.listdir(PICTURES_DIR)
                       if d.startswith("S") and os.path.isdir(os.path.join(PICTURES_DIR, d))])
    total = 0
    for i, subj in enumerate(subjects):
        data[subj] = {}
        for expr in EXPR_TYPES:
            fdir = os.path.join(PICTURES_DIR, subj, expr, "Colour frames")
            if not os.path.exists(fdir):
                continue
            paths = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
            if not paths:
                continue
            aus = []
            for j in range(0, len(paths), BATCH_SIZE):
                aus.append(extract_batch(model, dev, paths[j:j+BATCH_SIZE]))
            data[subj][expr] = np.concatenate(aus)
            total += len(paths)
        if (i+1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{len(subjects)} subjects ({total} frames, {time.time()-t0:.1f}s)")
    print(f"  ✅ Extracted AUs from {total} frames in {time.time()-t0:.1f}s")
    return data

def run_experiments(au_data):
    results = defaultdict(list)
    per_subj = []
    for subj, exprs in au_data.items():
        if "Neutral" not in exprs:
            continue
        neutral = exprs["Neutral"]
        mean_b, std_b = np.mean(neutral, 0), np.maximum(np.std(neutral, 0), 1e-4)
        w = np.ones(neutral.shape[1]); 
        for idx in PAIN_IDX: w[idx] = 3.0
        w /= w.sum()

        def baud_score(aus):
            z = np.maximum((aus - mean_b) / std_b, 0)
            raw = np.array([np.dot(w, f) for f in z])
            return float(np.mean(1.0 / (1.0 + np.exp(-raw + 2.0))))

        def gen_score(aus):
            return float(np.mean([np.mean(f[PAIN_IDX]) for f in aus]))

        def pspi(aus):
            s = [min((f[2]+max(f[4],f[5])+max(f[6],f[7])+(f[17] if len(f)>17 else 0))/2,1) for f in aus]
            return float(np.mean(s))

        results["BAUD (Ours)"].append({"score": baud_score(neutral), "true": 0})
        results["Generic"].append({"score": gen_score(neutral), "true": 0})
        results["PSPI"].append({"score": pspi(neutral), "true": 0})

        for expr in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            if expr not in exprs: continue
            pain = exprs[expr]
            results["BAUD (Ours)"].append({"score": baud_score(pain), "true": 1})
            results["Generic"].append({"score": gen_score(pain), "true": 1})
            results["PSPI"].append({"score": pspi(pain), "true": 1})
            per_subj.append({"subject": subj, "expression": expr,
                             "baud": baud_score(pain), "generic": gen_score(pain)})
    return results, per_subj

def compute_metrics(results):
    metrics = {}
    for name, entries in results.items():
        scores = [e["score"] for e in entries]
        truths = [e["true"] for e in entries]
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(truths, [1 if s > t else 0 for s in scores], zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        preds = [1 if s > best_t else 0 for s in scores]
        try: auc = roc_auc_score(truths, scores)
        except: auc = 0
        metrics[name] = {"acc": accuracy_score(truths, preds), "f1": best_f1,
                         "auc": auc, "thresh": best_t}
    return metrics

def main():
    print("=" * 60)
    print("  BAUD × PEMF × OpenGraphAU — Full Pipeline")
    print("=" * 60)

    model, dev = load_model()
    au_data = extract_all(model, dev)

    # Cache extracted AUs
    cache = {f"{s}_{e}": v for s, exprs in au_data.items() for e, v in exprs.items()}
    np.savez(os.path.join(RESULTS_DIR, "pemf_extracted_aus.npz"), **cache)
    print(f"  💾 Cached AUs to {RESULTS_DIR}/pemf_extracted_aus.npz")

    results, per_subj = run_experiments(au_data)
    metrics = compute_metrics(results)

    print("\n" + "=" * 70)
    print("  RESULTS: Real Face Images → OpenGraphAU → BAUD")
    print("=" * 70)
    print(f"  {'Method':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Thresh':>8}")
    print("-" * 70)
    for m, v in metrics.items():
        print(f"  {m:<25} {v['acc']:>8.4f} {v['f1']:>8.4f} {v['auc']:>8.4f} {v['thresh']:>8.2f}")
    print("=" * 70)

    with open(os.path.join(RESULTS_DIR, "opengraphau_metrics.txt"), "w") as f:
        for m, v in metrics.items():
            f.write(f"{m:<25} {v['acc']:>8.4f} {v['f1']:>8.4f} {v['auc']:>8.4f}\n")

    print("\n  ✅ COMPLETE. Run scripts/run_meta_learning.py next!")

if __name__ == "__main__":
    main()
