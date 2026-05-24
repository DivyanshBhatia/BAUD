"""
Baseline methods for comparison with BAUD.
"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PAIN_AU_INDICES, PAIN_AU_NAMES


class GenericPainDetector:
    """
    Non-personalized pain detection.
    Uses fixed thresholds on raw AU values — same for every patient.
    This is what most existing systems do.
    """

    def __init__(self, pain_au_indices=None):
        self.pain_au_indices = pain_au_indices or PAIN_AU_INDICES
        self.name = "Generic (no personalization)"

    def process_sequence(self, au_matrix: np.ndarray) -> List[float]:
        scores = []
        for frame in au_matrix:
            pain_aus = frame[self.pain_au_indices]
            score = float(np.mean(pain_aus))
            scores.append(score)
        return scores


class PSPICalculator:
    """
    Pitting Spontaneous Pain Intensity (PSPI) formula.
    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
    Standard clinical pain scoring from AUs.
    """

    def __init__(self):
        self.name = "PSPI Formula"

    def process_sequence(self, au_matrix: np.ndarray) -> List[float]:
        scores = []
        for frame in au_matrix:
            au4 = frame[2]               # AU4
            max_67 = max(frame[4], frame[5])   # max(AU6, AU7)
            max_910 = max(frame[6], frame[7])  # max(AU9, AU10)
            au43 = frame[17] if frame.shape[0] > 17 else 0  # AU43

            pspi = au4 + max_67 + max_910 + au43
            # Normalize to [0, 1]
            score = float(min(pspi / 4.0, 1.0))
            scores.append(score)
        return scores


class OneClassSVMBaseline:
    """
    One-class SVM anomaly detection on AU baseline.
    Learns "normal" from baseline, detects pain as anomalies.
    This is a personalized baseline but uses a simpler method than BAUD.
    """

    def __init__(self, nu=0.1, kernel="rbf", gamma="scale"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.is_fitted = False
        self.name = "One-Class SVM"

    def calibrate(self, baseline_aus: np.ndarray):
        pain_cols = PAIN_AU_INDICES
        features = baseline_aus[:, pain_cols]
        self.model.fit(features)
        self.is_fitted = True

    def process_sequence(self, au_matrix: np.ndarray) -> List[float]:
        assert self.is_fitted, "Call calibrate() first"
        pain_cols = PAIN_AU_INDICES
        features = au_matrix[:, pain_cols]
        # decision_function: positive = normal, negative = anomaly
        raw = self.model.decision_function(features)
        # Convert to pain score: more negative = more pain
        scores = [float(1.0 / (1.0 + np.exp(r))) for r in raw]
        return scores


class IsolationForestBaseline:
    """
    Isolation Forest anomaly detection on AU baseline.
    """

    def __init__(self, contamination=0.1, n_estimators=100, seed=42):
        self.model = IsolationForest(
            contamination=contamination, n_estimators=n_estimators,
            random_state=seed
        )
        self.is_fitted = False
        self.name = "Isolation Forest"

    def calibrate(self, baseline_aus: np.ndarray):
        pain_cols = PAIN_AU_INDICES
        self.model.fit(baseline_aus[:, pain_cols])
        self.is_fitted = True

    def process_sequence(self, au_matrix: np.ndarray) -> List[float]:
        assert self.is_fitted
        pain_cols = PAIN_AU_INDICES
        raw = self.model.decision_function(au_matrix[:, pain_cols])
        scores = [float(1.0 / (1.0 + np.exp(r))) for r in raw]
        return scores


class MahalanobisBaseline:
    """
    Mahalanobis distance from baseline centroid.
    Personalized, no learned weights — pure statistical deviation.
    This is an ablation of BAUD without the learned weighting.
    """

    def __init__(self):
        self.mean = None
        self.cov_inv = None
        self.is_fitted = False
        self.name = "Mahalanobis Distance"

    def calibrate(self, baseline_aus: np.ndarray):
        pain_cols = PAIN_AU_INDICES
        features = baseline_aus[:, pain_cols]
        self.mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        # Regularize for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        self.is_fitted = True

    def process_sequence(self, au_matrix: np.ndarray) -> List[float]:
        assert self.is_fitted
        pain_cols = PAIN_AU_INDICES
        features = au_matrix[:, pain_cols]

        scores = []
        for frame in features:
            diff = frame - self.mean
            dist = float(np.sqrt(diff @ self.cov_inv @ diff))
            # Normalize with sigmoid
            score = float(1.0 / (1.0 + np.exp(-dist + 3.0)))
            scores.append(score)
        return scores


def get_all_baselines() -> List:
    """Return instances of all baseline methods."""
    return [
        GenericPainDetector(),
        PSPICalculator(),
        OneClassSVMBaseline(),
        IsolationForestBaseline(),
        MahalanobisBaseline(),
    ]
