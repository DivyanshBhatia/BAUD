# BAUD: Baseline-Anchored AU Deviation Scoring

Code for *"When Does Zero-Label Personalization Help? A Three-Protocol Evaluation of Baseline-Calibrated Facial Pain Detection"* (FG 2027 submission, Paper ID 327).

---

## What This Repository Does

BAUD is a 43-parameter scorer that sits on top of a frozen AU encoder and asks: *does this patient's face right now deviate from how it looked at rest?*
No pain labels from the target patient are ever used.

This repository reproduces every number in the paper, including:

- **Three-protocol evaluation** (P1 label-informed disjoint, P2 label-blind first-K, P3 heuristic-assisted) across three datasets and 179 subjects
- **Mechanism decomposition** (centering vs. scaling vs. weighting)
- **Shuffled-baseline control** proving personalization is genuinely personal
- **Shrinkage parameter cross-validation** (LOSO, 22/24 folds select lambda=0.5)
- **Encoder-independence test** with a second AU backbone (Py-Feat)
- **Operating-point analysis** (PR-AUC and precision at 90% recall)

---

## Requirements

```
Python >= 3.10
PyTorch >= 1.13
numpy, scipy, scikit-learn, opencv-python, Pillow
```

Optional (encoder-independence test only):
```
py-feat >= 2.0
```

GPU recommended. All experiments run on a single GPU; AU extraction takes ~5 min per dataset on an A100, and all scoring experiments complete in under 2 min from cached AUs.

---

## Setup

### 1. Install

```bash
git clone [ANONYMOUS_URL]
cd BAUD
pip install -r requirements.txt
```

### 2. AU encoder

```bash
git clone https://github.com/CVI-SZU/ME-GraphAU.git external/OpenGraphAU
# Download OpenGprahAU-ResNet50_second_stage.pth into external/OpenGraphAU/checkpoints/
```

Then prepare the ResNet-50 backbone weights:

```python
import torch, os
from torchvision.models import resnet50, ResNet50_Weights
os.makedirs("external/OpenGraphAU/pretrain_models", exist_ok=True)
torch.save(resnet50(weights=ResNet50_Weights.DEFAULT).state_dict(),
           "external/OpenGraphAU/pretrain_models/resnet50-19c8e357.pth")
```

**AU ordering note.** OpenGraphAU outputs 27 main AUs + 14 bilateral sub-AUs = 41 dimensions.
The six prior-weighted AUs at indices `[2, 4, 5, 6, 7, 17]` are AU4, AU6, AU7, AU9, AU10 and AU20.
AU43 (eyes closed) is not in OpenGraphAU's output; AU20 (lip stretcher) takes its place in the prior.
The weighting ablation (Section V-B) shows this choice is immaterial.

### 3. Datasets

All datasets require their own data-use agreements:

| Dataset | Subjects | Pain type | Protocols |
|---------|----------|-----------|-----------|
| UNBC-McMaster | 25 (24 retained) | Clinical shoulder | P1, P2, P3 |
| PEMF | 68 | Pressure, laser, posed | P1 only |
| BioVid Part A | 87 | Graded heat (L0 vs L4) | P1, P2 (interleaved) |
| BioVid Part D | 90 | 7 posed expressions | Specificity analysis |

Expected layout:
```
data/
  UNBC/
    Images/Images/           # FACS-coded frames
    Frame_Labels/Frame_Labels/PSPI/
  PEMF/
    pictures/Pictures/       # Expression clips
    PEMF_Database.xlsx       # Metadata
  BioVid/
    PartA/                   # Heat-pain videos
    PartD/                   # Multi-expression videos
```

---

## Reproducing Results

### Step 1: Extract AUs (once per dataset)

```bash
python scripts/run_unbc.py          # Extracts + caches UNBC AUs
python scripts/run_pemf_opengraphau.py   # Extracts + caches PEMF AUs
python scripts/run_biovid.py        # Extracts + caches BioVid AUs
```

Cached `.npz` files are saved to `results/` and reused by all subsequent scripts.

### Step 2: Main experiments

| Script | What it produces | Paper section |
|--------|-----------------|---------------|
| `scripts/run_fg_fixes.py` | P1 disjoint, lambda CV, weighting ablation | V-A, V-B, V-D |
| `scripts/run_fg_framework.py` | Cross-dataset P1/P2/P3 table, deployment gap, pooled significance | V-A, V-C |
| `scripts/run_pyfeat_encoder.py` | Encoder-independence test (Py-Feat) | V-E |
| `scripts/run_fg_fixes_v2.py` | Bootstrap CIs, Sup. LR on matched eval sets, operating points | V-A, V-C |

### Step 3: Supplementary experiments

| Script | What it produces |
|--------|-----------------|
| `scripts/run_mechanism.py` | Shuffled-baseline control, centering decomposition |
| `scripts/run_partD_specificity.py` | Pain specificity across 7 expressions |
| `scripts/run_supplementary.py` | K-sweep, heuristic comparison, AUPRC, oracle bound, abstention |
| `scripts/run_significance.py` | Per-dataset and pooled Wilcoxon tests |

### Recommended run order (Colab)

```bash
# 1. Setup (~2 min)
pip install -r requirements.txt
# Download OpenGraphAU checkpoint and ResNet-50 backbone (see Setup)

# 2. AU extraction (~15 min total on A100, or use cached .npz)
python scripts/run_unbc.py

# 3. Core experiments (~5 min from cache)
python scripts/run_fg_fixes.py
python scripts/run_fg_framework.py
python scripts/run_fg_fixes_v2.py

# 4. Encoder independence (~12 min, requires py-feat)
pip install py-feat
python scripts/run_pyfeat_encoder.py
```

---

## Key Results

### P1 (label-informed, disjoint) — frame-level AUC

| Method | UNBC (n=24) | PEMF (n=68) | BioVid (n=87) |
|--------|-------------|-------------|---------------|
| BAUD (z-score) | 0.772 [.716,.826] | **0.975** | **0.598** |
| Center-only | **0.776** [.707,.830] | 0.953 | 0.594 |
| Generic | 0.699 [.613,.774] | 0.851 | 0.561 |
| Shuffled (control) | 0.645 [.594,.697] | 0.824 | 0.510 |

Pooled BAUD vs Generic: p < 10^-8 (179 subjects).

### Deployment gap (P1 to P2)

| Dataset | P1 | P2 | Gap | P2 + Shrinkage |
|---------|----|----|-----|----------------|
| UNBC | 0.772 | 0.680 | -9.2 pp | 0.718 |
| BioVid (interleaved) | 0.598 | 0.549 | -4.9 pp | 0.531 |

### Mechanism decomposition (UNBC, P1 disjoint)

| Component | AUC | Contribution |
|-----------|-----|-------------|
| Prior center-only | 0.776 | Reference (total +7.7 pp over Generic) |
| Uniform center-only | 0.755 | Centering: ~73% |
| Prior z-score | 0.772 | Weighting: ~27% |
| Learned z-score | 0.772 | Prior ~ Learned (p = 0.90) |
| Uniform z-score | 0.725 | Prior > Uniform (p = 0.003) |

### Encoder independence (UNBC)

| Finding | OpenGraphAU | Py-Feat | Replicates? |
|---------|-------------|---------|-------------|
| Own > Shuffled | p < 10^-4 | p < 10^-5 | Yes |
| Center ~ Full z | p = 0.73 | p = 0.66 | Yes |
| P2 erases gain | 0.772 -> 0.680 | 0.929 -> 0.787 | Yes |

---

## Hyperparameters

All values fixed on PEMF validation subjects before UNBC/BioVid evaluation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 50 | First-K calibration window |
| W | 10 | Low-activity sliding window (P3) |
| lambda | 0.5 | Shrinkage weight (LOSO-CV validated) |
| epsilon | 10^-4 | Minimum std floor |
| Pain AUs | AU4,6,7,9,10,20 | 3x prior weight (indices 2,4,5,6,7,17) |

---

## File Structure

```
BAUD/
  scripts/
    run_fg_fixes.py            # P1 disjoint, lambda CV, weighting ablation
    run_fg_framework.py        # Cross-dataset framework
    run_fg_fixes_v2.py         # CIs, matched Sup. LR, operating points
    run_pyfeat_encoder.py      # Encoder-independence test
    run_mechanism.py           # Shuffled baseline + centering decomposition
    run_partD_specificity.py   # Pain specificity (Part D)
    run_supplementary.py       # Supplementary experiments
    run_significance.py        # Statistical tests
    run_unbc.py                # UNBC AU extraction
    run_biovid.py              # BioVid AU extraction
    run_pemf_opengraphau.py    # PEMF AU extraction
  external/
    OpenGraphAU/               # Frozen AU encoder (not included; see Setup)
  results/                     # Cached AU vectors and experiment outputs
  config.py                    # Shared constants
  requirements.txt
  README.md
```

---

## Citation

```bibtex
@inproceedings{anonymous2027baud,
  title  = {When Does Zero-Label Personalization Help? A Three-Protocol
            Evaluation of Baseline-Calibrated Facial Pain Detection},
  author = {Anonymous},
  booktitle = {IEEE International Conference on Automatic Face and
               Gesture Recognition (FG)},
  year   = {2027}
}
```

## License

Released for research purposes. Dataset access requires separate agreements with the respective data providers.
