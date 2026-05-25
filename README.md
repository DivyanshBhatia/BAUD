# BAUD: Baseline-Anchored AU Deviation
## Personalized Pain Detection via Zero-Label Calibration in AU Space

> **AAAI-27 Target** | Abstracts due July 21, 2026 | Full papers due July 28, 2026

### Key Idea

Every person shows pain differently. BAUD watches a patient's resting face
(no labels needed), learns their personal AU baseline, then detects pain as
statistically significant deviations from *their* normal.

```
Face Image → [OpenGraphAU] → AU Vector → [Baseline Calibration] → Z-Scores → [Learned Weights] → Pain Score
                (frozen)                     (unlabeled)                        (meta-trained)
```

### Current Results (PEMF Dataset, 68 subjects, OpenGraphAU-extracted AUs)

| Method | Acc | F1 | AUC | Personalized? | Labels/Patient |
|--------|-----|-----|-----|---------------|----------------|
| **BAUD-Meta (Ours)** | **TBD** | **TBD** | **TBD** | Yes | Zero |
| BAUD-Prior | 0.9887 | 0.9924 | 0.9848 | Yes | Zero |
| Mahalanobis | 0.9925 | 0.9949 | 0.9995 | Yes | Zero |
| One-Class SVM | 0.9925 | 0.9949 | 0.9900 | Yes | Zero |
| Isolation Forest | 0.9662 | 0.9767 | 0.9958 | Yes | Zero |
| Generic | 0.7707 | 0.8585 | 0.8254 | No | N/A |
| PSPI | 0.7444 | 0.8534 | 0.5000 | No | N/A |

**Key finding:** ALL personalized methods (97-99%) crush ALL generic methods (50-83% AUC).
BAUD-Meta (learned weights) is expected to outperform statistical baselines.

---

## Quick Start on Google Colab

### 1. Setup
```python
# Upload baud-project-v4.tar.gz to Colab, then:
!tar -xzf baud-project-v4.tar.gz
!pip install -q scikit-learn matplotlib openpyxl timm

# Clone OpenGraphAU
!git clone https://github.com/lingjivoo/OpenGraphAU.git /content/baud/external/OpenGraphAU

# Download ResNet-50 pretrained weights (required by OpenGraphAU)
import os, torch
from torchvision.models import resnet50, ResNet50_Weights
os.makedirs("pretrain_models", exist_ok=True)
torch.save(resnet50(weights=ResNet50_Weights.DEFAULT).state_dict(),
           "pretrain_models/resnet50-19c8e357.pth")

# Place OpenGraphAU checkpoint at:
# /content/baud/external/OpenGraphAU/checkpoints/OpenGprahAU-ResNet50_second_stage.pth
```

### 2. Mount PEMF Data
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/pemf
!cp /content/drive/MyDrive/BAUD/pemf/PEMF_Database.xlsx /content/pemf/
!unzip -q /content/drive/MyDrive/BAUD/pemf/Pictures.zip -d /content/pemf/pictures/
```

### 3. Extract AUs (first time only — results are cached)
```python
!cd /content/baud && python scripts/run_pemf_opengraphau.py
```

### 4. Run Meta-Learning (the main experiment)
```python
!cd /content/baud && python scripts/run_meta_learning.py
```

### 5. View Results
```python
from IPython.display import Image, display
display(Image("/content/results/meta_training_curve.png"))
display(Image("/content/results/meta_final_comparison.png"))
!cat /content/results/meta_learning_results.txt
```

---

## Project Structure

```
baud/
├── config.py                          # Hyperparameters and paths
├── models/
│   ├── baud.py                        # Numpy BAUD calibrator (prior weights)
│   └── baud_learnable.py              # PyTorch meta-learnable BAUD
├── scripts/
│   ├── run_meta_learning.py           # ★ Main experiment: meta-learn weights
│   ├── run_pemf_opengraphau.py        # Full pipeline: images → AUs → BAUD
│   └── run_pemf_experiment.py         # FACS annotations experiment
├── data/
│   └── pemf_loader.py                 # PEMF dataset loader
├── baselines/
│   └── methods.py                     # All baseline methods
├── utils/
│   ├── metrics.py                     # Evaluation metrics
│   └── visualization.py               # Plotting utilities
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Status | Purpose |
|---------|--------|---------|
| PEMF | ✅ Downloaded | Prototype + first results (68 subjects) |
| BioVid | ⏳ Request sent | Primary paper dataset (90 subjects, 4 pain levels) |
| X-ITE | ⏳ Request sent | Cross-dataset generalization (134 subjects) |

## Citation

```bibtex
@article{baud2026,
  title={Pain is Personal: Zero-Label Patient Calibration for Personalized
         Pain Detection via Baseline-Anchored AU Deviation},
  author={Abhishek G and Raghu Vishnu Jalneela and Dr. Mehala N},
  year={2026}
}
```
