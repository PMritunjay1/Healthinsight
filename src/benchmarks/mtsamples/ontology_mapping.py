"""
MTSamples Ontology Mapping Module
=================================
Implements conservative, evidence-based clinical ontology mapping:
1. Maps ground-truth diagnostic sections, sample names, descriptions, and keywords
   to the 18 target disease categories across the 3 frozen specialists.
2. Applies strict negative and negation filters (e.g. "no evidence of infarction").
3. Marks ambiguous or non-authoritative candidate records explicitly as UNMAPPED.
4. Preserves negative control cases with zero active specialist diseases to measure specificity.
"""

import os
import sys
import re
import logging
from typing import Dict, List, Set, Any, Tuple
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("MTSamplesOntologyMapping")

# Specialist Disease Class Taxonomies (Exact Frozen Names)
CARDIOLOGIST_CLASSES = [
    "angina",
    "arrhythmia",
    "atrial fibrillation",
    "coronary artery disease",
    "heart failure",
    "hypertension",
]

PULMONOLOGIST_CLASSES = [
    "asthma",
    "copd",
    "pneumonia",
    "respiratory_failure",
    "pleural_effusion",
    "pulmonary_embolism",
]

NEUROLOGIST_CLASSES = [
    "stroke_ischemic",
    "intracranial_hemorrhage",
    "epilepsy_seizures",
    "altered_mental_status_encephalopathy",
    "transient_ischemic_attack",
    "neuropathy_neurodegenerative",
]

# Strict Regex Patterns for Authoritative Ground Truth Mapping
CARDIOLOGIST_RULES = {
    "angina": [
        r"\bangina\b",
        r"\bangina pectoris\b",
        r"\bunstable angina\b",
        r"\bstable angina\b",
        r"\bprinzmetal\b",
        r"\banginal symptoms\b",
    ],
    "arrhythmia": [
        r"\barrhythmia\b",
        r"\bdysrhythmia\b",
        r"\bventricular tachycardia\b",
        r"\bvtach\b",
        r"\bsupraventricular tachycardia\b",
        r"\bsvt\b",
        r"\bbradycardia\b",
        r"\bsinus bradycardia\b",
        r"\bsick sinus syndrome\b",
        r"\bheart block\b",
        r"\bcomplete heart block\b",
        r"\bav block\b",
        r"\bbundle branch block\b",
        r"\bpremature ventricular contractions\b",
        r"\bpvc\b",
        r"\bwpw syndrome\b",
        r"\bwolff-parkinson-white\b",
    ],
    "atrial fibrillation": [
        r"\batrial fibrillation\b",
        r"\bafib\b",
        r"\ba-fib\b",
        r"\batrial flutter\b",
        r"\baflutter\b",
    ],
    "coronary artery disease": [
        r"\bcoronary artery disease\b",
        r"\bcad\b",
        r"\bcoronary heart disease\b",
        r"\bmyocardial infarction\b",
        r"\bstemi\b",
        r"\bnstemi\b",
        r"\bacute coronary syndrome\b",
        r"\bacs\b",
        r"\bcoronary atherosclerosis\b",
        r"\batherosclerotic heart disease\b",
        r"\bashd\b",
        r"\bischemic cardiomyopathy\b",
        r"\bcoronary stenosis\b",
        r"\bcoronary artery stenosis\b",
    ],
    "heart failure": [
        r"\bheart failure\b",
        r"\bcongestive heart failure\b",
        r"\bchf\b",
        r"\bcardiomyopathy\b",
        r"\bsystolic heart failure\b",
        r"\bdiastolic heart failure\b",
        r"\bacute decompensated heart failure\b",
        r"\bcongestive cardiomyopathy\b",
        r"\bdilated cardiomyopathy\b",
    ],
    "hypertension": [
        r"\bhypertension\b",
        r"\bhigh blood pressure\b",
        r"\bessential hypertension\b",
        r"\bsystemic hypertension\b",
        r"\bhtn\b",
        r"\bhypertensive urgency\b",
        r"\bhypertensive emergency\b",
        r"\bhypertensive cardiovascular disease\b",
    ],
}

PULMONOLOGIST_RULES = {
    "asthma": [
        r"\basthma\b",
        r"\bbronchial asthma\b",
        r"\bstatus asthmaticus\b",
        r"\basthma exacerbation\b",
        r"\bacute asthma\b",
        r"\breactive airway disease\b",
    ],
    "copd": [
        r"\bcopd\b",
        r"\bchronic obstructive pulmonary disease\b",
        r"\bemphysema\b",
        r"\bchronic bronchitis\b",
        r"\bcopd exacerbation\b",
        r"\bacute exacerbation of copd\b",
    ],
    "pneumonia": [
        r"\bpneumonia\b",
        r"\bbacterial pneumonia\b",
        r"\bviral pneumonia\b",
        r"\baspiration pneumonia\b",
        r"\blobar pneumonia\b",
        r"\bcommunity-acquired pneumonia\b",
        r"\bcommunity acquired pneumonia\b",
        r"\bcap\b",
        r"\bhospital-acquired pneumonia\b",
        r"\bhap\b",
        r"\blung consolidation\b",
    ],
    "respiratory_failure": [
        r"\brespiratory failure\b",
        r"\bacute respiratory failure\b",
        r"\bhypoxemic respiratory failure\b",
        r"\bhypercapnic respiratory failure\b",
        r"\bacute respiratory distress syndrome\b",
        r"\bards\b",
        r"\bventilatory failure\b",
    ],
    "pleural_effusion": [
        r"\bpleural effusion\b",
        r"\bempyema\b",
        r"\bparapneumonic effusion\b",
        r"\bpleural fluid collection\b",
        r"\bhydrothorax\b",
        r"\bhemothorax\b",
    ],
    "pulmonary_embolism": [
        r"\bpulmonary embolism\b",
        r"\bpulmonary embolus\b",
        r"\bpe\b",
        r"\bsaddle embolus\b",
        r"\bpulmonary thromboembolism\b",
        r"\bpulmonary infarction\b",
    ],
}

NEUROLOGIST_RULES = {
    "stroke_ischemic": [
        r"\bischemic stroke\b",
        r"\bcerebral infarction\b",
        r"\bacute ischemic stroke\b",
        r"\bcerebrovascular accident\b",
        r"\bcva\b",
        r"\bacute cva\b",
        r"\bmca stroke\b",
        r"\bmca infarction\b",
        r"\blacunar stroke\b",
        r"\blacunar infarct\b",
        r"\bcerebral ischemia\b",
    ],
    "intracranial_hemorrhage": [
        r"\bintracranial hemorrhage\b",
        r"\bintracerebral hemorrhage\b",
        r"\bsubarachnoid hemorrhage\b",
        r"\bich\b",
        r"\bsah\b",
        r"\bsubdural hematoma\b",
        r"\bsdh\b",
        r"\bepidural hematoma\b",
        r"\bedh\b",
        r"\bhemorrhagic stroke\b",
        r"\bintraventricular hemorrhage\b",
        r"\bivh\b",
    ],
    "epilepsy_seizures": [
        r"\bepilepsy\b",
        r"\bseizure\b",
        r"\bseizures\b",
        r"\bseizure disorder\b",
        r"\bstatus epilepticus\b",
        r"\bconvulsion\b",
        r"\bconvulsions\b",
        r"\btonic-clonic seizure\b",
        r"\bgrand mal seizure\b",
        r"\bfocal seizure\b",
        r"\bepileptic encephalopathy\b",
    ],
    "altered_mental_status_encephalopathy": [
        r"\baltered mental status\b",
        r"\bams\b",
        r"\bencephalopathy\b",
        r"\btoxic metabolic encephalopathy\b",
        r"\bmetabolic encephalopathy\b",
        r"\bhepatic encephalopathy\b",
        r"\bdelirium\b",
        r"\bacute confusion\b",
        r"\bacute confusional state\b",
        r"\bstupor\b",
        r"\bcoma\b",
    ],
    "transient_ischemic_attack": [
        r"\btransient ischemic attack\b",
        r"\btia\b",
        r"\bamaurosis fugax\b",
        r"\btransient cerebral ischemia\b",
    ],
    "neuropathy_neurodegenerative": [
        r"\bneuropathy\b",
        r"\bperipheral neuropathy\b",
        r"\bpolyneuropathy\b",
        r"\bdiabetic neuropathy\b",
        r"\bparkinson\b",
        r"\bparkinson's disease\b",
        r"\bparkinson's\b",
        r"\bparkinsonism\b",
        r"\balzheimer\b",
        r"\balzheimer's disease\b",
        r"\balzheimer's\b",
        r"\bdementia\b",
        r"\bvascular dementia\b",
        r"\bmultiple sclerosis\b",
        r"\bms\b",
        r"\bamyotrophic lateral sclerosis\b",
        r"\bals\b",
        r"\bmyasthenia gravis\b",
        r"\bguillain-barre\b",
        r"\bguillain barre\b",
        r"\bradiculopathy\b",
        r"\bcervical radiculopathy\b",
        r"\blumbar radiculopathy\b",
    ],
}

# Negation Patterns to prevent false positives in ground-truth text
NEGATION_PATTERNS = [
    r"no evidence of\s+([a-z\s]+)",
    r"rule out\s+([a-z\s]+)",
    r"r/o\s+([a-z\s]+)",
    r"negative for\s+([a-z\s]+)",
    r"denies\s+([a-z\s]+)",
    r"without evidence of\s+([a-z\s]+)",
    r"free of\s+([a-z\s]+)",
]


def match_disease_rules(
    text: str, rules_dict: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    """
    Match disease rules against diagnostic text, checking for negations.
    Returns:
        Tuple of (positive_matches, negated_matches)
    """
    text_lower = text.lower()
    positives = []
    negated = []

    for disease_name, patterns in rules_dict.items():
        matched = False
        for pat in patterns:
            for m in re.finditer(pat, text_lower):
                start_span = max(0, m.start() - 30)
                context = text_lower[start_span:m.end()]
                # Check if negated
                is_neg = False
                for neg_pat in NEGATION_PATTERNS:
                    if re.search(neg_pat, context):
                        is_neg = True
                        break

                if is_neg:
                    negated.append(disease_name)
                else:
                    positives.append(disease_name)
                    matched = True
                    break
            if matched:
                break

    # If both positive and negated, remove from positives
    final_positives = [p for p in set(positives) if p not in set(negated)]
    return sorted(list(final_positives)), sorted(list(set(negated)))


def map_mtsamples_ontology(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Map all records in MTSamples DataFrame into multi-label ground truth vectors for:
    - Cardiologist Specialist (6 classes)
    - Pulmonologist Specialist (6 classes)
    - Neurologist Specialist (6 classes)
    """
    mapped_records = []

    cardio_counts = {c: 0 for c in CARDIOLOGIST_CLASSES}
    pulmo_counts = {c: 0 for c in PULMONOLOGIST_CLASSES}
    neuro_counts = {c: 0 for c in NEUROLOGIST_CLASSES}

    total_records = len(df)
    cardio_positive_records = 0
    pulmo_positive_records = 0
    neuro_positive_records = 0
    multispecialist_overlap_records = 0
    negative_control_records = 0
    unmapped_records = 0
    ambiguous_records = 0

    for idx, row in df.iterrows():
        # Build diagnostic evidence text strictly from ground truth / diagnostic sources
        diag_source = f"{row.get('ground_truth_diagnostic_text', '')} {row.get('sample_name', '')} {row.get('description', '')} {row.get('keywords', '')}".strip()
        spec_category = str(row.get("medical_specialty", "")).strip()

        cardio_pos, cardio_neg = match_disease_rules(diag_source, CARDIOLOGIST_RULES)
        pulmo_pos, pulmo_neg = match_disease_rules(diag_source, PULMONOLOGIST_RULES)
        neuro_pos, neuro_neg = match_disease_rules(diag_source, NEUROLOGIST_RULES)

        has_cardio = len(cardio_pos) > 0
        has_pulmo = len(pulmo_pos) > 0
        has_neuro = len(neuro_pos) > 0

        # Construct multi-label binary ground truth vectors
        cardio_vector = [1 if c in cardio_pos else 0 for c in CARDIOLOGIST_CLASSES]
        pulmo_vector = [1 if c in pulmo_pos else 0 for c in PULMONOLOGIST_CLASSES]
        neuro_vector = [1 if c in neuro_pos else 0 for c in NEUROLOGIST_CLASSES]

        # Categorize mapping status
        active_specialties = []
        if has_cardio:
            active_specialties.append("Cardiology")
            cardio_positive_records += 1
            for d in cardio_pos:
                cardio_counts[d] += 1

        if has_pulmo:
            active_specialties.append("Pulmonology")
            pulmo_positive_records += 1
            for d in pulmo_pos:
                pulmo_counts[d] += 1

        if has_neuro:
            active_specialties.append("Neurology")
            neuro_positive_records += 1
            for d in neuro_pos:
                neuro_counts[d] += 1

        if len(active_specialties) > 1:
            multispecialist_overlap_records += 1

        # Determine if case is a known negative control or truly unmapped
        is_mapped = has_cardio or has_pulmo or has_neuro
        if is_mapped:
            mapping_status = "MAPPED_SPECIALIST_POSITIVE"
        else:
            # Check if it belongs to a clear non-specialist specialty (negative control)
            if spec_category in [
                "Surgery", "Orthopedic", "Gastroenterology", "Urology",
                "Obstetrics / Gynecology", "ENT - Otolaryngology", "Ophthalmology",
                "Dermatology", "Dentistry", "Podiatry", "Pediatrics - Neonatal",
                "Hematology - Oncology", "Nephrology", "Bariatrics", "Rheumatology"
            ]:
                mapping_status = "NEGATIVE_CONTROL_INACTIVE"
                negative_control_records += 1
            else:
                # Ambiguous consult or general note with no identifiable condition
                mapping_status = "UNMAPPED_AMBIGUOUS"
                unmapped_records += 1
                ambiguous_records += 1

        rec = row.to_dict()
        rec["cardio_gt_labels"] = cardio_pos
        rec["cardio_gt_vector"] = cardio_vector
        rec["pulmo_gt_labels"] = pulmo_pos
        rec["pulmo_gt_vector"] = pulmo_vector
        rec["neuro_gt_labels"] = neuro_pos
        rec["neuro_gt_vector"] = neuro_vector
        rec["active_gt_specialties"] = active_specialties
        rec["mapping_status"] = mapping_status
        rec["is_specialist_positive"] = is_mapped
        rec["cardio_negated"] = cardio_neg
        rec["pulmo_negated"] = pulmo_neg
        rec["neuro_negated"] = neuro_neg

        mapped_records.append(rec)

    mapped_df = pd.DataFrame(mapped_records)

    total_eligible_benchmark = cardio_positive_records + pulmo_positive_records + neuro_positive_records + negative_control_records
    mapping_percentage = (total_eligible_benchmark / total_records) * 100.0

    stats = {
        "total_records": total_records,
        "cardio_positive_records": cardio_positive_records,
        "pulmo_positive_records": pulmo_positive_records,
        "neuro_positive_records": neuro_positive_records,
        "multispecialist_overlap_records": multispecialist_overlap_records,
        "negative_control_records": negative_control_records,
        "unmapped_ambiguous_records": unmapped_records,
        "total_eligible_benchmark_records": total_eligible_benchmark,
        "mapping_coverage_pct": round(mapping_percentage, 2),
        "cardio_disease_counts": cardio_counts,
        "pulmo_disease_counts": pulmo_counts,
        "neuro_disease_counts": neuro_counts,
    }

    logger.info(
        f"Ontology Mapping Complete: {total_eligible_benchmark}/{total_records} ({mapping_percentage:.2f}% eligible). "
        f"Cardio: {cardio_positive_records}, Pulmo: {pulmo_positive_records}, Neuro: {neuro_positive_records}, "
        f"Negative Controls: {negative_control_records}, Unmapped: {unmapped_records}"
    )

    return mapped_df, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.benchmarks.mtsamples.load_mtsamples import load_raw_mtsamples
    from src.benchmarks.mtsamples.preprocess_mtsamples import preprocess_all_mtsamples
    df, _ = load_raw_mtsamples()
    proc_df, _ = preprocess_all_mtsamples(df)
    mapped_df, stats = map_mtsamples_ontology(proc_df)
    print("--- Ontology Mapping Statistics ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
