"""
MTSamples Preprocessing & Pre-Diagnostic Boundary Module
=========================================================
Implements conservative pre-diagnostic text extraction:
1. Segments authentic clinical transcriptions into clinical sections.
2. Preserves ONLY pre-diagnostic clinical presentation (Chief Complaint, HPI, PMH, ROS, Physical Exam, Vitals).
3. Strictly removes post-diagnostic sections (Diagnosis, Impression, Postoperative Diagnosis, Procedure, Assessment/Plan).
4. Records exactly what sections were extracted vs removed for auditing.
"""

import os
import sys
import re
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("MTSamplesPreprocessor")

# Recognized Pre-Diagnostic Section Headers (Case-insensitive matching)
PRE_DIAGNOSTIC_HEADERS = [
    r"CHIEF COMPLAINT",
    r"REASON FOR VISIT",
    r"REASON FOR CONSULTATION",
    r"REASON FOR ADMISSION",
    r"HISTORY OF PRESENT ILLNESS",
    r"HPI",
    r"PAST MEDICAL HISTORY",
    r"PMH",
    r"PAST SURGICAL HISTORY",
    r"PSH",
    r"PAST HISTORY",
    r"MEDICAL HISTORY",
    r"REVIEW OF SYSTEMS",
    r"ROS",
    r"PHYSICAL EXAMINATION",
    r"EXAMINATION",
    r"PHYSICAL EXAM",
    r"PE",
    r"VITAL SIGNS",
    r"VITALS",
    r"SUBJECTIVE",
    r"OBJECTIVE",
    r"CURRENT MEDICATIONS",
    r"MEDICATIONS",
    r"ALLERGIES",
    r"SOCIAL HISTORY",
    r"FAMILY HISTORY",
    r"FAMILY AND SOCIAL HISTORY",
    r"EMERGENCY DEPARTMENT COURSE",
    r"LABORATORY DATA",
    r"LABS",
    r"GENERAL",
    r"HEENT",
    r"CHEST",
    r"LUNGS",
    r"CARDIOVASCULAR",
    r"HEART",
    r"ABDOMEN",
    r"EXTREMITIES",
    r"NEUROLOGICAL",
    r"NEUROLOGIC EXAM",
    r"CONSTITUTIONAL",
    r"ADMISSION HISTORY",
    r"INDICATION FOR SURGERY",
    r"INDICATION FOR PROCEDURE",
    r"INDICATIONS FOR OPERATION",
]

# Post-Diagnostic / Procedural Headers to strictly strip from input text
POST_DIAGNOSTIC_HEADERS = [
    r".*PRE[\s\-_]?(?:OPERATIVE|PROCEDUR\w*)\s+DIAGNOS.*",
    r".*POST[\s\-_]?(?:OPERATIVE|PROCEDUR\w*)\s+DIAGNOS.*",
    r".*FINAL\s+DIAGNOS.*",
    r".*DISCHARGE\s+DIAGNOS.*",
    r".*ADMITTING\s+DIAGNOS.*",
    r".*PRIMARY\s+DIAGNOS.*",
    r".*SECONDARY\s+DIAGNOS.*",
    r".*DIAGNOS(?:IS|ES|TIC).*",
    r".*IMPRESSION.*",
    r".*ASSESSMENT.*",
    r".*PLAN.*",
    r".*RECOMMENDATION.*",
    r".*PROCEDURE.*",
    r".*TITLE OF OPERATION.*",
    r".*NAME OF OPERATION.*",
    r".*OPERATION.*",
    r".*SURGERY.*",
    r".*OPERATIVE FINDINGS.*",
    r".*FINDINGS.*",
    r".*COMPLICATIONS.*",
    r".*DISPOSITION.*",
    r".*DISCHARGE INSTRUCTIONS.*",
    r".*HOSPITAL COURSE.*",
]


def extract_sections(transcription: str) -> List[Tuple[str, str]]:
    """
    Parse a transcription note into a sequence of (header_title, section_content).
    Handles newline, comma, semicolon, or sentence-delimited UPPERCASE_HEADER: patterns.
    """
    if not transcription or not isinstance(transcription, str):
        return []

    # Regex finding UPPERCASE_HEADER: patterns preceded by start of line, newline, or punctuation
    header_pattern = re.compile(r"(?:^|\n|\r|[,\.\;]\s*)([A-Z0-9\s/\-_&]{3,50}):\s*", re.MULTILINE)
    matches = list(header_pattern.finditer(transcription))

    if not matches:
        return [("UNSTRUCTURED", transcription.strip())]

    sections = []
    # If text exists before first header
    if matches[0].start() > 0:
        lead_text = transcription[:matches[0].start()].strip(" ,;\n\r")
        if lead_text:
            sections.append(("HEADERLESS_PREAMBLE", lead_text))

    for i, match in enumerate(matches):
        header_name = match.group(1).strip()
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(transcription)
        content = transcription[start_pos:end_pos].strip(" ,;\n\r")
        sections.append((header_name, content))

    return sections


def is_header_match(header: str, pattern_list: List[str]) -> bool:
    """Check if header matches any regex in pattern list."""
    h_clean = header.upper().strip()
    for p in pattern_list:
        if re.search(r"^" + p + r"$", h_clean, re.IGNORECASE):
            return True
        if re.search(r"\b" + p + r"\b", h_clean, re.IGNORECASE):
            return True
    return False


def scrub_diagnostic_leakage(text: str) -> Tuple[str, List[str]]:
    """
    Post-processing scrub: Iteratively detect and truncate any post-diagnostic headers/sections
    that might have survived initial extraction until none remain.
    """
    scrubbed_terms = []
    text_out = text

    # Universal diagnostic/procedural header leak pattern
    leak_pat = re.compile(
        r"(?:^|\n|\r|[,\.\;]\s*)([A-Z0-9\s/\-_&]*?(?:DIAGNOS|IMPRESSION|ASSESSMENT|OPERATION|PROCEDURE|SURGERY|DISPOSITION|COMPLICATION|HOSPITAL\s+COURSE|DISCHARGE\s+INSTRUCTION)[A-Z0-9\s/\-_&]*?:)",
        re.IGNORECASE
    )

    while True:
        match = leak_pat.search(text_out)
        if not match:
            break
        term_name = match.group(1).upper().strip(" :,\n\r")
        scrubbed_terms.append(term_name)
        # Truncate strictly at match position
        text_out = text_out[:match.start()].strip(" ,;\n\r")

    return text_out, scrubbed_terms


def preprocess_clinical_record(row: pd.Series) -> Dict[str, Any]:
    """
    Process a single MTSamples record:
    1. Extracts pre-diagnostic text strictly.
    2. Isolates ground-truth diagnostic sections for label verification.
    3. Scrubs any accidental diagnostic residual.
    4. Records section extraction audit info.
    """
    raw_text = str(row.get("transcription", "")).strip()
    sections = extract_sections(raw_text)

    prediagnostic_parts = []
    postdiagnostic_parts = []
    sections_extracted = []
    sections_removed = []

    for header, content in sections:
        if not content:
            continue

        h_upper = header.upper()
        # Strictly check POST-DIAGNOSTIC headers FIRST to prevent misclassification
        if (
            is_header_match(header, POST_DIAGNOSTIC_HEADERS) or
            any(k in h_upper for k in ["DIAGNOS", "IMPRESSION", "ASSESSMENT", "PROCEDURE", "OPERATION", "SURGERY", "FINDING", "PLAN", "COMPLICATION", "DISPOSITION"])
        ):
            postdiagnostic_parts.append(f"{header}: {content}")
            sections_removed.append(header)
        elif is_header_match(header, PRE_DIAGNOSTIC_HEADERS):
            clean_content, scrubbed = scrub_diagnostic_leakage(content)
            if clean_content:
                prediagnostic_parts.append(f"{header}: {clean_content}")
                sections_extracted.append(header)
            if scrubbed:
                sections_removed.extend(scrubbed)
        elif header in ["HEADERLESS_PREAMBLE", "UNSTRUCTURED"]:
            clean_content, scrubbed = scrub_diagnostic_leakage(content)
            if clean_content:
                prediagnostic_parts.append(clean_content)
                sections_extracted.append("PREAMBLE_EXTRACTED")
            if scrubbed:
                sections_removed.extend(scrubbed)
        else:
            # Ambiguous / Secondary section (e.g. ANESTHESIA, ESTIMATED BLOOD LOSS, IMPLANT)
            # Default to removing to guarantee ZERO look-ahead leak
            sections_removed.append(header)

    # Combine parts and run final scrub pass across full text
    if not prediagnostic_parts:
        safe_lead = raw_text[:500]
        safe_lead, scrubbed = scrub_diagnostic_leakage(safe_lead)
        prediagnostic_text = safe_lead if safe_lead else "Patient presented for clinical evaluation."
        extraction_mode = "FALLBACK_SCRUBBED_LEAD"
    else:
        prediagnostic_text = "\n\n".join(prediagnostic_parts).strip()
        prediagnostic_text, scrubbed = scrub_diagnostic_leakage(prediagnostic_text)
        if scrubbed:
            sections_removed.extend(scrubbed)
        extraction_mode = "STRUCTURED_SECTION_EXTRACTION"

    # Final guarantee pass against any residual diagnostic headers
    for term in ["DIAGNOSIS:", "FINAL DIAGNOSIS:", "DISCHARGE DIAGNOSIS:", "PREOPERATIVE DIAGNOSIS:", "POSTOPERATIVE DIAGNOSIS:", "IMPRESSION:", "IMPRESSIONS:", "ASSESSMENT/PLAN:"]:
        idx = prediagnostic_text.upper().find(term)
        if idx != -1:
            prediagnostic_text = prediagnostic_text[:idx].strip(" ,;\n\r")

    if not prediagnostic_text:
        prediagnostic_text = "Patient presented for clinical evaluation."

    ground_truth_text = "\n\n".join(postdiagnostic_parts).strip()
    if not ground_truth_text:
        # Fallback to sample_name + description + keywords if note lacked explicit diagnosis section
        ground_truth_text = f"{row.get('sample_name', '')} {row.get('description', '')} {row.get('keywords', '')}".strip()

    return {
        "prediagnostic_text": prediagnostic_text,
        "ground_truth_diagnostic_text": ground_truth_text,
        "extraction_mode": extraction_mode,
        "sections_extracted": sections_extracted,
        "sections_removed": sections_removed,
        "num_prediagnostic_chars": len(prediagnostic_text),
        "num_raw_chars": len(raw_text)
    }


def preprocess_all_mtsamples(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply pre-diagnostic text preprocessing to entire MTSamples dataset.
    """
    records = []
    extraction_modes = {}
    total_sections_removed = 0
    total_sections_extracted = 0

    for idx, row in df.iterrows():
        proc = preprocess_clinical_record(row)
        proc_row = row.to_dict()
        proc_row["clean_prediagnostic_text"] = proc["prediagnostic_text"]
        proc_row["ground_truth_diagnostic_text"] = proc["ground_truth_diagnostic_text"]
        proc_row["extraction_mode"] = proc["extraction_mode"]
        proc_row["sections_extracted"] = proc["sections_extracted"]
        proc_row["sections_removed"] = proc["sections_removed"]
        proc_row["prediagnostic_len"] = proc["num_prediagnostic_chars"]
        records.append(proc_row)

        m = proc["extraction_mode"]
        extraction_modes[m] = extraction_modes.get(m, 0) + 1
        total_sections_extracted += len(proc["sections_extracted"])
        total_sections_removed += len(proc["sections_removed"])

    processed_df = pd.DataFrame(records)

    stats = {
        "total_records_processed": len(processed_df),
        "extraction_mode_breakdown": extraction_modes,
        "total_pre_diagnostic_sections_extracted": total_sections_extracted,
        "total_post_diagnostic_sections_removed": total_sections_removed,
        "mean_prediagnostic_chars": float(processed_df["prediagnostic_len"].mean()),
        "min_prediagnostic_chars": int(processed_df["prediagnostic_len"].min()),
        "max_prediagnostic_chars": int(processed_df["prediagnostic_len"].max())
    }

    logger.info(
        f"Preprocessed {len(processed_df)} records. "
        f"Modes: {extraction_modes}. Sections removed: {total_sections_removed}"
    )

    return processed_df, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.benchmarks.mtsamples.load_mtsamples import load_raw_mtsamples
    df, _ = load_raw_mtsamples()
    proc_df, stats = preprocess_all_mtsamples(df)
    print("--- Preprocessing Statistics ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
    sample = proc_df.iloc[0]
    print("\nSample 0 Clean Pre-Diagnostic Text Preview:")
    print(sample["clean_prediagnostic_text"][:300])
