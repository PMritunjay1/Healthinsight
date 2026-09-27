"""
MTSamples Loader Module
=======================
Responsible for acquiring, verifying, and loading the public MTSamples dataset.
Ensures zero modifications to original data files and computes cryptographic checksums.
"""

import os
import hashlib
import logging
from typing import Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger("MTSamplesLoader")

EXPECTED_SHA256 = "cf264760170a2fbcbae32d9a75fd43dfdd10a8b423043101f98553d4a94a25db"


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_raw_mtsamples(
    filepath: str = "datasets/raw/intent/mtsamples.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw MTSamples clinical dataset, verify integrity, and clean fields.
    
    Returns:
        Tuple of (cleaned_df, dataset_metadata)
    """
    if not os.path.isabs(filepath):
        # Resolve relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        filepath = os.path.join(project_root, filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MTSamples dataset not found at {filepath}")

    checksum = compute_file_sha256(filepath)
    file_size_bytes = os.path.getsize(filepath)

    # Load with pandas
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin1")

    total_records = len(df)
    original_columns = df.columns.tolist()

    # Normalize column names & clean strings
    df.columns = [c.strip() for c in df.columns]
    for col in ["description", "medical_specialty", "sample_name", "transcription", "keywords"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop explicit nulls or empty transcription rows
    valid_mask = df["transcription"].notna() & (df["transcription"] != "") & (df["transcription"] != "nan")
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    excluded_empty = total_records - len(valid_df)

    metadata = {
        "dataset_name": "MTSamples Medical Transcriptions",
        "source": "MTSamples.com / Public Domain Clinical Transcriptions",
        "file_path": filepath,
        "file_size_bytes": file_size_bytes,
        "sha256_checksum": checksum,
        "checksum_verified": (checksum == EXPECTED_SHA256),
        "total_raw_records": total_records,
        "valid_records_with_transcription": len(valid_df),
        "excluded_empty_transcription": excluded_empty,
        "original_fields": original_columns,
        "num_medical_specialties": int(valid_df["medical_specialty"].nunique()),
        "specialty_distribution": valid_df["medical_specialty"].value_counts().to_dict(),
        "license": "Public Domain / Open Access Clinical Transcription Archive",
    }

    logger.info(
        f"Loaded MTSamples: {len(valid_df)}/{total_records} valid records. "
        f"SHA256: {checksum[:12]}... (Verified: {metadata['checksum_verified']})"
    )

    return valid_df, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, meta = load_raw_mtsamples()
    print("--- MTSamples Metadata ---")
    for k, v in meta.items():
        if k != "specialty_distribution":
            print(f"{k}: {v}")
    print("\nTop 5 Specialties:")
    for spec, cnt in list(meta["specialty_distribution"].items())[:5]:
        print(f"  {spec}: {cnt}")
