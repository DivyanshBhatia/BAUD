"""
Synthetic patient data generator for BAUD pipeline testing.
Generates realistic AU sequences for stoic and expressive patient types.
"""
import numpy as np
from typing import Dict, List, Tuple


# Pain-related AU boost magnitudes (how much each AU increases during pain)
PAIN_AU_BOOSTS = {
    2: 0.15,   # AU4 (brow lowerer) — strong pain indicator
    4: 0.10,   # AU6 (cheek raise)
    5: 0.12,   # AU7 (lid tightener)
    6: 0.08,   # AU9 (nose wrinkle)
    7: 0.07,   # AU10 (upper lip raiser)
    17: 0.14,  # AU43 (eye closure) — strong pain indicator
}

PATIENT_PROFILES = {
    "stoic": {
        "base_mean_range": (0.02, 0.12),
        "base_std_range": (0.01, 0.03),
        "description": "Low resting AU activation, minimal variation",
    },
    "expressive": {
        "base_mean_range": (0.15, 0.45),
        "base_std_range": (0.04, 0.12),
        "description": "High resting AU activation, large variation",
    },
    "moderate": {
        "base_mean_range": (0.08, 0.25),
        "base_std_range": (0.02, 0.06),
        "description": "Moderate resting AU activation",
    },
    "random": {
        "base_mean_range": (0.02, 0.40),
        "base_std_range": (0.01, 0.10),
        "description": "Random profile for diversity",
    },
}


def generate_patient(
    patient_type: str = "stoic",
    num_baseline: int = 100,
    num_pain: int = 80,
    num_aus: int = 41,
    seed: int = 42,
) -> Dict:
    """
    Generate synthetic AU data for one patient.

    Returns dict with:
        - baseline_aus: (num_baseline, num_aus)
        - pain_aus: (num_pain, num_aus)
        - pain_labels: (num_pain,) — 0=none, 1=mild, 2=moderate, 3=severe
        - metadata: patient info
    """
    rng = np.random.RandomState(seed)

    profile = PATIENT_PROFILES.get(patient_type, PATIENT_PROFILES["random"])
    lo_mean, hi_mean = profile["base_mean_range"]
    lo_std, hi_std = profile["base_std_range"]

    # Generate patient-specific baseline AU profile
    base_mean = rng.uniform(lo_mean, hi_mean, size=num_aus)
    base_std = rng.uniform(lo_std, hi_std, size=num_aus)

    # --- Baseline frames (resting, no pain) ---
    baseline_aus = rng.normal(
        loc=base_mean, scale=base_std, size=(num_baseline, num_aus)
    ).clip(0, 1)

    # --- Pain frames ---
    pain_aus = rng.normal(
        loc=base_mean, scale=base_std, size=(num_pain, num_aus)
    ).clip(0, 1)

    # Create pain labels: ramp from no-pain to severe
    pain_labels = np.zeros(num_pain, dtype=int)
    seg = num_pain // 4
    pain_labels[seg : 2 * seg] = 1        # mild
    pain_labels[2 * seg : 3 * seg] = 2    # moderate
    pain_labels[3 * seg :] = 3            # severe

    # Add pain signal to pain-related AUs
    for frame_idx in range(num_pain):
        intensity = pain_labels[frame_idx]
        for au_idx, boost in PAIN_AU_BOOSTS.items():
            if au_idx < num_aus:
                noise = 1.0 + rng.normal(0, 0.2)
                pain_aus[frame_idx, au_idx] += boost * intensity * max(noise, 0.3)
    pain_aus = pain_aus.clip(0, 1)

    return {
        "baseline_aus": baseline_aus,
        "pain_aus": pain_aus,
        "pain_labels": pain_labels,
        "metadata": {
            "patient_type": patient_type,
            "seed": seed,
            "base_mean": base_mean,
            "base_std": base_std,
            "num_baseline": num_baseline,
            "num_pain": num_pain,
        },
    }


def generate_cohort(
    num_patients: int = 20,
    num_baseline: int = 100,
    num_pain: int = 80,
    num_aus: int = 41,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate a cohort of diverse synthetic patients.
    Mixes stoic, expressive, moderate, and random types.
    """
    rng = np.random.RandomState(seed)
    types = ["stoic", "expressive", "moderate", "random"]

    cohort = []
    for i in range(num_patients):
        ptype = types[i % len(types)]
        patient_seed = seed + i * 1000
        patient = generate_patient(
            patient_type=ptype,
            num_baseline=num_baseline,
            num_pain=num_pain,
            num_aus=num_aus,
            seed=patient_seed,
        )
        patient["patient_id"] = i
        cohort.append(patient)

    print(f"Generated cohort of {num_patients} patients:")
    for t in types:
        count = sum(1 for p in cohort if p["metadata"]["patient_type"] == t)
        print(f"  {t}: {count} patients")

    return cohort


def split_cohort(
    cohort: List[Dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split cohort into meta-train, meta-val, meta-test."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(cohort))

    n_train = int(len(cohort) * train_ratio)
    n_val = int(len(cohort) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train = [cohort[i] for i in train_idx]
    val = [cohort[i] for i in val_idx]
    test = [cohort[i] for i in test_idx]

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
