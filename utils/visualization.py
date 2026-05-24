"""
Visualization utilities for BAUD results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PAIN_AU_INDICES, PAIN_AU_NAMES, RESULTS_DIR


def plot_pain_scores_comparison(
    results_dict,
    labels,
    patient_name,
    save_path=None,
):
    """
    Plot pain scores over time for multiple methods.

    Args:
        results_dict: {method_name: [pain_scores]}
        labels: ground truth pain labels
        patient_name: string for title
        save_path: where to save
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#795548"]

    frames = range(len(labels))

    # Shade ground truth
    for i, label in enumerate(labels):
        if label > 0:
            alpha = label / 3 * 0.15
            ax.axvspan(i - 0.5, i + 0.5, alpha=alpha, color="#4CAF50")

    for idx, (name, scores) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        style = "-" if idx == 0 else "--"
        lw = 2.5 if idx == 0 else 1.5
        ax.plot(frames, scores, color=color, linewidth=lw,
                label=name, linestyle=style, alpha=0.9)

    ax.set_title(f"Pain Detection — {patient_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Pain Score", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_au_deviation_heatmap(
    z_scores_dict,
    save_path=None,
):
    """
    Plot per-AU z-score heatmaps for multiple patients side by side.

    Args:
        z_scores_dict: {patient_name: z_scores_array (frames, num_aus)}
    """
    n = len(z_scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, z_scores) in zip(axes, z_scores_dict.items()):
        pain_z = z_scores[:, PAIN_AU_INDICES].T
        im = ax.imshow(pain_z, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=5, interpolation="nearest")
        ax.set_yticks(range(len(PAIN_AU_NAMES)))
        ax.set_yticklabels(PAIN_AU_NAMES, fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_title(f"{name} — Per-AU Z-Scores", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Z-Score (σ)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_calibration_ablation(
    durations,
    f1_scores,
    auc_scores,
    save_path=None,
):
    """
    Plot performance vs calibration duration.

    Args:
        durations: list of calibration frame counts
        f1_scores: corresponding F1 scores
        auc_scores: corresponding AUC scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(durations, f1_scores, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.set_xlabel("Calibration Frames", fontsize=12)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("F1 vs Calibration Duration", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(durations, auc_scores, "s-", color="#FF5722", linewidth=2, markersize=8)
    ax2.set_xlabel("Calibration Frames", fontsize=12)
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_title("AUC vs Calibration Duration", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_personalization_comparison(
    pain_aus_A, pain_aus_B, z_scores_A, z_scores_B,
    frame_idx=70,
    save_path=None,
):
    """
    The 'money plot' showing why personalization matters.
    Same AU values → different z-scores for different patients.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(PAIN_AU_NAMES))
    width = 0.35

    # Raw AU values
    aus_A = pain_aus_A[frame_idx, PAIN_AU_INDICES]
    aus_B = pain_aus_B[frame_idx, PAIN_AU_INDICES]
    ax1.bar(x - width / 2, aus_A, width, label="Patient A (stoic)",
            color="#1976D2", alpha=0.8)
    ax1.bar(x + width / 2, aus_B, width, label="Patient B (expressive)",
            color="#E64A19", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(PAIN_AU_NAMES)
    ax1.set_ylabel("Raw AU Value")
    ax1.set_title(f"Raw AU Values at Frame {frame_idx}", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Personalized z-scores
    z_A = np.maximum(z_scores_A[frame_idx, PAIN_AU_INDICES], 0)
    z_B = np.maximum(z_scores_B[frame_idx, PAIN_AU_INDICES], 0)
    ax2.bar(x - width / 2, z_A, width, label="Patient A (stoic)",
            color="#1976D2", alpha=0.8)
    ax2.bar(x + width / 2, z_B, width, label="Patient B (expressive)",
            color="#E64A19", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(PAIN_AU_NAMES)
    ax2.set_ylabel("Z-Score (σ from baseline)")
    ax2.set_title(f"BAUD Personalized Z-Scores at Frame {frame_idx}",
                  fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(y=2, color="red", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def generate_clinical_report(reports, pain_scores, labels, patient_name,
                              frame_idx=None) -> str:
    """Generate a text-based clinical AU deviation report."""
    if frame_idx is None:
        frame_idx = int(np.argmax(pain_scores))

    report = reports[frame_idx]
    score = pain_scores[frame_idx]
    true_label = labels[frame_idx]
    label_names = {0: "No Pain", 1: "Mild", 2: "Moderate", 3: "Severe"}

    if score < 0.3:
        pred_name = "No Pain"
    elif score < 0.5:
        pred_name = "Mild"
    elif score < 0.7:
        pred_name = "Moderate"
    else:
        pred_name = "Severe"

    lines = []
    lines.append("=" * 55)
    lines.append(f"  BAUD Clinical Report — {patient_name}")
    lines.append(f"  Frame: {frame_idx}")
    lines.append("=" * 55)
    lines.append(f"  Pain Score:      {score:.3f}")
    lines.append(f"  Predicted Level: {pred_name}")
    lines.append(f"  Ground Truth:    {label_names.get(true_label, 'Unknown')}")
    lines.append("-" * 55)
    lines.append("  Per-AU Deviation from Patient Baseline:")
    lines.append("-" * 55)

    sorted_aus = sorted(report.items(), key=lambda x: x[1]["z_score"], reverse=True)
    for au_name, data in sorted_aus:
        z = data["z_score"]
        bar_len = int(min(abs(z), 10))
        bar = "█" * bar_len
        flag = " ⚠️  ALERT" if z > 2.0 else ""
        lines.append(
            f"  {au_name:>6s}: {z:>+6.2f}σ  {bar:<10s} "
            f"(val={data['value']:.3f}, base={data['baseline_mean']:.3f})"
            f"{flag}"
        )
    lines.append("=" * 55)

    return "\n".join(lines)
