"""
PEMF Dataset Loader for BAUD Pipeline
======================================
Loads the Pain E-Motion Faces Database and maps it to BAUD's
calibration (neutral) + test (pain) structure.

Expected folder structure on Colab:
/content/pemf/
├── PEMF_Database.xlsx
├── pictures/Pictures/Modified/{Subject}/{ExprType}/Colour frames/*.jpg
└── clips/Clips/Modified/{Subject}/{ExprType}/...

Usage:
    from pemf_loader import PEMFDataset
    dataset = PEMFDataset("/content/pemf")
    dataset.load()
    subjects = dataset.get_subject_splits()
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Expression type mapping
EXPR_TYPES = {
    "Neutral": {"code": "N", "is_pain": False, "pain_level": 0},
    "Algometer Pain": {"code": "A", "is_pain": True, "pain_level": 1},
    "Laser Pain": {"code": "L", "is_pain": True, "pain_level": 2},
    "Posed Pain": {"code": "P", "is_pain": True, "pain_level": 3},
}

# AU columns in PEMF_Database.xlsx
AU_COLUMNS = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU12", "AU20",
              "AU25", "AU26", "AU27", "AU43", "AU45"]


class PEMFDataset:
    """
    PEMF dataset loader for BAUD experiments.

    Key mapping for BAUD:
    - Neutral frames → calibration baseline (unlabeled)
    - Algometer/Laser/Posed Pain frames → pain detection targets
    """

    def __init__(self, root_dir: str = "/content/pemf"):
        self.root_dir = root_dir
        self.pictures_dir = os.path.join(root_dir, "pictures", "Pictures", "Modified")
        self.clips_dir = os.path.join(root_dir, "clips", "Clips", "Modified")
        self.xlsx_path = os.path.join(root_dir, "PEMF_Database.xlsx")

        self.metadata = None       # DataFrame from xlsx
        self.subjects = []         # List of subject IDs (S001-S068)
        self.subject_data = {}     # {subject_id: {expr_type: [frame_paths]}}
        self.au_annotations = {}   # {clip_code: {au_name: value}}

    def load(self):
        """Load metadata and discover all frame paths."""
        print("=" * 60)
        print("  Loading PEMF Dataset")
        print("=" * 60)

        # Load metadata xlsx
        self._load_metadata()

        # Discover subjects and frame paths
        self._discover_frames()

        # Parse AU annotations from metadata
        self._parse_au_annotations()

        print(f"\n  Total subjects: {len(self.subjects)}")
        print(f"  Subjects with all 4 expression types: "
              f"{self._count_complete_subjects()}")
        print("  ✅ PEMF dataset loaded!")
        print("=" * 60)

    def _load_metadata(self):
        """Load and parse PEMF_Database.xlsx."""
        if not os.path.exists(self.xlsx_path):
            print(f"  ⚠️  {self.xlsx_path} not found. Continuing without metadata.")
            self.metadata = None
            return

        self.metadata = pd.read_excel(self.xlsx_path)
        print(f"  Loaded metadata: {len(self.metadata)} entries")
        print(f"  Columns: {list(self.metadata.columns)}")

    def _discover_frames(self):
        """Find all frame image paths organized by subject and expression."""
        if not os.path.exists(self.pictures_dir):
            raise FileNotFoundError(
                f"Pictures directory not found: {self.pictures_dir}\n"
                f"Expected structure: {self.root_dir}/pictures/Pictures/Modified/"
            )

        # Find all subject folders
        subject_dirs = sorted([
            d for d in os.listdir(self.pictures_dir)
            if os.path.isdir(os.path.join(self.pictures_dir, d))
            and d.startswith("S")
        ])
        self.subjects = subject_dirs

        print(f"\n  Found {len(subject_dirs)} subjects")

        # For each subject, find frames per expression type
        for subj in subject_dirs:
            self.subject_data[subj] = {}
            subj_dir = os.path.join(self.pictures_dir, subj)

            for expr_name, expr_info in EXPR_TYPES.items():
                expr_dir = os.path.join(subj_dir, expr_name, "Colour frames")

                if os.path.exists(expr_dir):
                    frames = sorted(glob.glob(os.path.join(expr_dir, "*.jpg")))
                    if frames:
                        self.subject_data[subj][expr_name] = frames

            # Print summary for first few subjects
            if subj in subject_dirs[:3]:
                exprs = list(self.subject_data[subj].keys())
                counts = {e: len(self.subject_data[subj][e]) for e in exprs}
                print(f"    {subj}: {counts}")

        print(f"    ... ({len(subject_dirs) - 3} more subjects)")

    def _parse_au_annotations(self):
        """Extract AU annotations from metadata."""
        if self.metadata is None:
            return

        for _, row in self.metadata.iterrows():
            clip_code = str(row.get("Clip", ""))
            if not clip_code:
                continue

            au_values = {}
            for au_col in AU_COLUMNS:
                val = row.get(au_col, 0)
                try:
                    au_values[au_col] = int(val) if pd.notna(val) else 0
                except (ValueError, TypeError):
                    au_values[au_col] = 0

            self.au_annotations[clip_code] = au_values

        print(f"  Parsed AU annotations for {len(self.au_annotations)} clips")

    def _count_complete_subjects(self) -> int:
        """Count subjects that have all 4 expression types."""
        count = 0
        for subj, data in self.subject_data.items():
            if len(data) == 4:
                count += 1
        return count

    def get_subject_frames(
        self,
        subject_id: str,
        expression: str = "Neutral",
        max_frames: Optional[int] = None,
    ) -> List[str]:
        """
        Get frame paths for a subject and expression type.

        Args:
            subject_id: e.g., "S001"
            expression: "Neutral", "Algometer Pain", "Laser Pain", "Posed Pain"
            max_frames: limit number of frames

        Returns:
            List of image file paths
        """
        frames = self.subject_data.get(subject_id, {}).get(expression, [])
        if max_frames:
            frames = frames[:max_frames]
        return frames

    def get_subject_baseline_and_pain(
        self, subject_id: str
    ) -> Dict:
        """
        Get calibration (neutral) and pain frames for one subject.
        This is the key method for BAUD experiments.

        Returns:
            {
                "subject_id": str,
                "baseline_frames": [paths],   ← neutral frames for calibration
                "pain_frames": {
                    "Algometer Pain": [paths],
                    "Laser Pain": [paths],
                    "Posed Pain": [paths],
                },
                "au_annotations": {clip_code: {au: value}},
                "metadata": {age, gender, ...}
            }
        """
        data = self.subject_data.get(subject_id, {})
        result = {
            "subject_id": subject_id,
            "baseline_frames": data.get("Neutral", []),
            "pain_frames": {},
            "au_annotations": {},
            "metadata": {},
        }

        for expr_name in ["Algometer Pain", "Laser Pain", "Posed Pain"]:
            if expr_name in data:
                result["pain_frames"][expr_name] = data[expr_name]

        # Attach AU annotations
        subj_num = subject_id  # e.g., "S001"
        for code_suffix in ["N", "A", "L", "P"]:
            clip_code = f"{subj_num}{code_suffix}"
            if clip_code in self.au_annotations:
                result["au_annotations"][clip_code] = self.au_annotations[clip_code]

        # Attach metadata from xlsx
        if self.metadata is not None:
            subj_rows = self.metadata[
                self.metadata["Clip"].str.startswith(subject_id)
            ]
            if len(subj_rows) > 0:
                first_row = subj_rows.iloc[0]
                result["metadata"] = {
                    "age": first_row.get("Age", None),
                    "gender": first_row.get("Gender", None),
                }

        return result

    def get_subject_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        seed: int = 42,
        require_complete: bool = True,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split subjects into meta-train, meta-val, meta-test.

        Args:
            train_ratio: fraction for training
            val_ratio: fraction for validation
            seed: random seed
            require_complete: only use subjects with all 4 expression types

        Returns:
            (train_subjects, val_subjects, test_subjects)
        """
        if require_complete:
            subjects = [s for s in self.subjects
                        if len(self.subject_data.get(s, {})) == 4]
        else:
            subjects = [s for s in self.subjects
                        if "Neutral" in self.subject_data.get(s, {})]

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(subjects))

        n_train = int(len(subjects) * train_ratio)
        n_val = int(len(subjects) * val_ratio)

        train = [subjects[i] for i in indices[:n_train]]
        val = [subjects[i] for i in indices[n_train:n_train + n_val]]
        test = [subjects[i] for i in indices[n_train + n_val:]]

        print(f"Subject split: train={len(train)}, val={len(val)}, test={len(test)}")
        return sorted(train), sorted(val), sorted(test)

    def summary(self):
        """Print dataset summary."""
        total_frames = sum(
            len(frames)
            for subj_data in self.subject_data.values()
            for frames in subj_data.values()
        )
        print(f"\nPEMF Dataset Summary:")
        print(f"  Subjects: {len(self.subjects)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Expression types per subject:")
        for expr_name in EXPR_TYPES:
            count = sum(
                1 for s in self.subjects
                if expr_name in self.subject_data.get(s, {})
            )
            avg_frames = np.mean([
                len(self.subject_data[s][expr_name])
                for s in self.subjects
                if expr_name in self.subject_data.get(s, {})
            ]) if count > 0 else 0
            print(f"    {expr_name}: {count} subjects, ~{avg_frames:.0f} frames each")


# ============================================================================
# Convenience function: Load PEMF and prepare for BAUD
# ============================================================================

def load_pemf_for_baud(
    root_dir: str = "/content/pemf",
    seed: int = 42,
) -> Tuple[PEMFDataset, List[str], List[str], List[str]]:
    """
    One-liner to load PEMF and get train/val/test splits.

    Usage:
        dataset, train, val, test = load_pemf_for_baud("/content/pemf")
    """
    dataset = PEMFDataset(root_dir)
    dataset.load()
    dataset.summary()
    train, val, test = dataset.get_subject_splits(seed=seed)
    return dataset, train, val, test


if __name__ == "__main__":
    # Quick test
    dataset, train, val, test = load_pemf_for_baud("/content/pemf")

    # Show one subject's data
    sample = dataset.get_subject_baseline_and_pain(train[0])
    print(f"\nSample subject: {sample['subject_id']}")
    print(f"  Baseline frames: {len(sample['baseline_frames'])}")
    for pain_type, frames in sample["pain_frames"].items():
        print(f"  {pain_type} frames: {len(frames)}")
    print(f"  AU annotations: {sample['au_annotations']}")
    print(f"  Metadata: {sample['metadata']}")
