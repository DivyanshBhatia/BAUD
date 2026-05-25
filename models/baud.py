"""
BAUD: Baseline-Anchored AU Deviation (numpy version).
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


class BAUDCalibrator:
    """Zero-label personalized pain scorer using AU baseline deviation."""

    def __init__(self, num_aus=41, pain_indices=None, window=5,
                 pain_weight=3.0, epsilon=1e-4, sigmoid_shift=2.0):
        self.num_aus = num_aus
        self.pain_indices = pain_indices or [2, 4, 5, 6, 7, 17]
        self.window = window
        self.epsilon = epsilon
        self.sigmoid_shift = sigmoid_shift
        self.baseline_mean = None
        self.baseline_std = None
        self.is_calibrated = False
        self.z_buffer = []

        # Prior deviation weights
        self.weights = np.ones(num_aus)
        for idx in self.pain_indices:
            if idx < num_aus:
                self.weights[idx] = pain_weight
        self.weights /= self.weights.sum()

    def calibrate(self, baseline_aus: np.ndarray):
        """Calibrate from unlabeled baseline frames (n_frames, num_aus)."""
        self.baseline_mean = np.mean(baseline_aus, axis=0)
        self.baseline_std = np.maximum(np.std(baseline_aus, axis=0), self.epsilon)
        self.is_calibrated = True
        self.z_buffer = []

    def score_frame(self, au_vector: np.ndarray) -> Dict:
        assert self.is_calibrated
        z = (au_vector - self.baseline_mean) / self.baseline_std
        z_pos = np.maximum(z, 0)
        self.z_buffer.append(z_pos)
        if len(self.z_buffer) > self.window:
            self.z_buffer.pop(0)
        z_smooth = np.mean(self.z_buffer, axis=0)
        raw = np.dot(self.weights, z_smooth)
        score = 1.0 / (1.0 + np.exp(-raw + self.sigmoid_shift))
        return {"pain_score": float(score), "z_scores": z, "z_smoothed": z_smooth}

    def score_sequence(self, au_matrix: np.ndarray) -> Tuple[List[float], np.ndarray]:
        self.z_buffer = []
        scores, all_z = [], []
        for frame in au_matrix:
            r = self.score_frame(frame)
            scores.append(r["pain_score"])
            all_z.append(r["z_scores"])
        return scores, np.array(all_z)
