"""
Evaluation metrics for pain detection.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from typing import Dict, List


def binary_pain_metrics(
    pain_scores: List[float],
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate binary pain detection (pain vs no-pain)."""
    binary_truth = (ground_truth > 0).astype(int)
    binary_pred = (np.array(pain_scores) > threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(binary_truth, binary_pred),
        "f1": f1_score(binary_truth, binary_pred, zero_division=0),
        "precision": precision_score(binary_truth, binary_pred, zero_division=0),
        "recall": recall_score(binary_truth, binary_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(binary_truth, pain_scores)
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


def multilevel_pain_metrics(
    pain_scores: List[float],
    ground_truth: np.ndarray,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """Evaluate multi-level pain detection (4 levels)."""
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]  # Boundaries for 4 levels

    scores = np.array(pain_scores)
    pred_levels = np.zeros_like(scores, dtype=int)
    pred_levels[scores > thresholds[0]] = 1
    pred_levels[scores > thresholds[1]] = 2
    pred_levels[scores > thresholds[2]] = 3

    return {
        "accuracy_4level": accuracy_score(ground_truth, pred_levels),
        "f1_macro": f1_score(ground_truth, pred_levels, average="macro", zero_division=0),
        "f1_weighted": f1_score(ground_truth, pred_levels, average="weighted", zero_division=0),
    }


def find_optimal_threshold(
    pain_scores: List[float],
    ground_truth: np.ndarray,
) -> float:
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_thresh = 0.5
    binary_truth = (ground_truth > 0).astype(int)

    for thresh in np.arange(0.1, 0.9, 0.05):
        pred = (np.array(pain_scores) > thresh).astype(int)
        f1 = f1_score(binary_truth, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    title: str = "Pain Detection Results",
) -> str:
    """Format metrics as a printable table."""
    lines = []
    lines.append("=" * 75)
    lines.append(f"  {title}")
    lines.append("=" * 75)
    lines.append(
        f"  {'Method':<30} {'Acc':>8} {'F1':>8} {'Prec':>8} "
        f"{'Recall':>8} {'AUC':>8}"
    )
    lines.append("-" * 75)

    for method_name, metrics in results.items():
        lines.append(
            f"  {method_name:<30} "
            f"{metrics.get('accuracy', 0):.4f}  "
            f"{metrics.get('f1', 0):.4f}  "
            f"{metrics.get('precision', 0):.4f}  "
            f"{metrics.get('recall', 0):.4f}  "
            f"{metrics.get('auc', 0):.4f}"
        )

    lines.append("=" * 75)
    return "\n".join(lines)
