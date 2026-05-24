"""
BAUD: Baseline-Anchored AU Deviation

Core module implementing:
1. Patient baseline calibration (zero labels)
2. Per-AU z-score deviation scoring
3. Hand-crafted deviation weighting (prior-based)
4. Temporal smoothing

No PyTorch dependency — runs on CPU with numpy.
For the meta-learnable version, see baud_learnable.py.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    NUM_AUS, PAIN_AU_INDICES, PAIN_AU_NAMES, PAIN_AU_PRIOR_WEIGHT,
    BASELINE_EPSILON, TEMPORAL_WINDOW_SIZE, PAIN_SIGMOID_SHIFT,
)


class BAUDCalibrator:
    """
    Baseline-Anchored AU Deviation calibrator.

    Use this for:
    - Quick testing without GPU
    - Baseline comparison (prior weights, no meta-learning)
    - Pipeline validation on synthetic/real data
    """

    def __init__(
        self,
        num_aus: int = NUM_AUS,
        pain_au_indices: List[int] = None,
        window_size: int = TEMPORAL_WINDOW_SIZE,
        pain_prior_weight: float = PAIN_AU_PRIOR_WEIGHT,
    ):
        self.num_aus = num_aus
        self.pain_au_indices = pain_au_indices or PAIN_AU_INDICES
        self.window_size = window_size

        # Patient baseline (populated during calibration)
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.is_calibrated = False

        # Temporal buffer
        self.z_score_buffer: List[np.ndarray] = []

        # Prior-based deviation weights
        self.deviation_weights = np.ones(num_aus)
        for idx in self.pain_au_indices:
            if idx < num_aus:
                self.deviation_weights[idx] = pain_prior_weight
        self.deviation_weights /= self.deviation_weights.sum()

    def calibrate(self, baseline_aus: np.ndarray) -> Dict:
        """
        Calibrate from unlabeled baseline AU data.

        Args:
            baseline_aus: (num_frames, num_aus) from patient's resting face

        Returns:
            dict with calibration stats
        """
        assert baseline_aus.ndim == 2
        assert baseline_aus.shape[1] == self.num_aus, (
            f"Expected {self.num_aus} AUs, got {baseline_aus.shape[1]}"
        )

        self.baseline_mean = np.mean(baseline_aus, axis=0)
        self.baseline_std = np.maximum(np.std(baseline_aus, axis=0), BASELINE_EPSILON)
        self.is_calibrated = True
        self.z_score_buffer = []

        return {
            "num_frames": len(baseline_aus),
            "mean": self.baseline_mean.copy(),
            "std": self.baseline_std.copy(),
        }

    def score_frame(self, au_vector: np.ndarray) -> Dict:
        """
        Score a single frame against patient baseline.

        Returns dict with:
            - pain_score: float in [0, 1]
            - z_scores: per-AU z-scores
            - z_smoothed: temporally smoothed z-scores
            - au_report: per-pain-AU deviation details
        """
        assert self.is_calibrated, "Call calibrate() first!"

        # Per-AU z-scores
        z_scores = (au_vector - self.baseline_mean) / self.baseline_std
        z_positive = np.maximum(z_scores, 0)  # Only positive deviations

        # Temporal smoothing
        self.z_score_buffer.append(z_positive)
        if len(self.z_score_buffer) > self.window_size:
            self.z_score_buffer.pop(0)
        z_smoothed = np.mean(self.z_score_buffer, axis=0)

        # Weighted pain score
        raw_score = np.dot(self.deviation_weights, z_smoothed)
        pain_score = float(1.0 / (1.0 + np.exp(-raw_score + PAIN_SIGMOID_SHIFT)))

        # Per-AU report for pain AUs
        au_report = {}
        for i, idx in enumerate(self.pain_au_indices):
            if idx < self.num_aus:
                au_report[PAIN_AU_NAMES[i]] = {
                    "value": float(au_vector[idx]),
                    "baseline_mean": float(self.baseline_mean[idx]),
                    "baseline_std": float(self.baseline_std[idx]),
                    "z_score": float(z_scores[idx]),
                    "z_smoothed": float(z_smoothed[idx]),
                }

        return {
            "pain_score": pain_score,
            "z_scores": z_scores,
            "z_smoothed": z_smoothed,
            "au_report": au_report,
        }

    def process_sequence(
        self, au_matrix: np.ndarray
    ) -> Tuple[List[float], np.ndarray, List[Dict]]:
        """
        Process a sequence of AU vectors.

        Args:
            au_matrix: (num_frames, num_aus)

        Returns:
            pain_scores, all_z_scores, all_reports
        """
        self.z_score_buffer = []  # Reset buffer

        pain_scores = []
        all_z_scores = []
        all_reports = []

        for frame_aus in au_matrix:
            result = self.score_frame(frame_aus)
            pain_scores.append(result["pain_score"])
            all_z_scores.append(result["z_scores"])
            all_reports.append(result["au_report"])

        return pain_scores, np.array(all_z_scores), all_reports

    def reset(self):
        """Reset calibration and buffers."""
        self.baseline_mean = None
        self.baseline_std = None
        self.is_calibrated = False
        self.z_score_buffer = []
