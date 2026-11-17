"""
HealthInsight-UQ Shared Label Extractor Utility
================================================
Provides authoritative mapping between medical specialties and their corresponding
dictionary keys in MIMIC-IV cohort datasets.

Corrects the historical evaluation defect where Pulmonology ('pulmo_disease_labels')
and Neurology ('neuro_disease_labels') were accessed using nonexistent dictionary keys
instead of 'pulmonary_disease_labels' and 'neurology_disease_labels'.
"""

from typing import Dict, List, Optional, Set

# Authoritative dictionary key mapping per specialty
LABEL_KEY_MAP: Dict[str, str] = {
    "Cardiology": "cardio_disease_labels",
    "Pulmonology": "pulmonary_disease_labels",
    "Neurology": "neurology_disease_labels",
}

# 18-disease canonical vocabulary across the 3 specialist silos (6 per specialty)
SPECIALTY_LABEL_MAP: Dict[str, List[str]] = {
    "Cardiology": [
        "hypertension",
        "heart failure",
        "coronary artery disease",
        "atrial fibrillation",
        "arrhythmia",
        "angina",
    ],
    "Pulmonology": [
        "asthma",
        "copd",
        "pneumonia",
        "respiratory_failure",
        "pleural_effusion",
        "pulmonary_embolism",
    ],
    "Neurology": [
        "neuropathy_neurodegenerative",
        "epilepsy_seizures",
        "altered_mental_status_encephalopathy",
        "stroke_ischemic",
        "intracranial_hemorrhage",
        "transient_ischemic_attack",
    ],
}

ALL_DISEASE_CLASSES: List[str] = (
    SPECIALTY_LABEL_MAP["Cardiology"]
    + SPECIALTY_LABEL_MAP["Pulmonology"]
    + SPECIALTY_LABEL_MAP["Neurology"]
)


def normalize_specialty(specialty: str) -> str:
    """
    Normalizes specialty name into one of {'Cardiology', 'Pulmonology', 'Neurology'}.
    Accepts abbreviations, lowercase, and role names ('cardiologist', etc.).
    """
    s = (specialty or "").strip().lower()
    if "cardio" in s:
        return "Cardiology"
    elif "pulmo" in s or "respirat" in s:
        return "Pulmonology"
    elif "neuro" in s:
        return "Neurology"
    else:
        raise ValueError(f"Unrecognized specialty domain: {specialty}")


def get_specialty_key(specialty: str) -> str:
    """Returns the exact dataset dictionary key for a given specialty."""
    canonical = normalize_specialty(specialty)
    return LABEL_KEY_MAP[canonical]


def get_allowed_labels(specialty: str) -> Set[str]:
    """Returns the set of valid canonical disease labels for a given specialty."""
    canonical = normalize_specialty(specialty)
    return set(SPECIALTY_LABEL_MAP[canonical])


def extract_ground_truth_labels(
    record: dict,
    specialty: Optional[str] = None,
    filter_valid: bool = True,
) -> List[str]:
    """
    Extracts, normalizes, and validates ground-truth disease labels from a cohort record.

    Parameters:
    -----------
    record: dict
        A record dictionary from test_split.json, train_split.json, or evaluation ledgers.
    specialty: Optional[str]
        The clinical specialty of the record. If omitted, checks 'specialty' key in record.
    filter_valid: bool
        If True, restricts labels to the canonical 6-disease vocabulary for the specialty.

    Returns:
    --------
    List[str]: Cleaned, lowercased, deduplicated list of disease labels.
    """
    if specialty is None:
        specialty = record.get("specialty")
    
    if not specialty:
        # Fallback inspection across possible dictionary keys
        for spec, key in LABEL_KEY_MAP.items():
            if key in record and record[key]:
                specialty = spec
                break
        if not specialty:
            # Check if ground_truth_labels already exists
            if "ground_truth_labels" in record and record["ground_truth_labels"]:
                raw = record["ground_truth_labels"]
                return sorted(list(set(str(x).lower().strip() for x in raw if str(x).strip())))
            return []

    canonical_spec = normalize_specialty(specialty)
    primary_key = LABEL_KEY_MAP[canonical_spec]

    raw_labels = record.get(primary_key)
    if raw_labels is None:
        # Check alternative or logged keys
        raw_labels = record.get("ground_truth_labels") or []

    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]

    cleaned = [str(x).lower().strip() for x in raw_labels if str(x).strip()]

    if filter_valid:
        allowed = get_allowed_labels(canonical_spec)
        cleaned = [x for x in cleaned if x in allowed]

    return sorted(list(set(cleaned)))
