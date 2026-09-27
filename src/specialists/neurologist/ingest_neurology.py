import os
import sys
import gzip
import csv
import re
import json
import collections
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

NEUROLOGY_ICD_TAXONOMY = {
    "stroke_ischemic": {
        "icd10": ["I63", "I65", "I66"],
        "icd9": ["433", "434", "436"]
    },
    "intracranial_hemorrhage": {
        "icd10": ["I60", "I61", "I62", "S06"],
        "icd9": ["430", "431", "432", "852", "853"]
    },
    "epilepsy_seizures": {
        "icd10": ["G40", "R56"],
        "icd9": ["345", "7803"]
    },
    "altered_mental_status_encephalopathy": {
        "icd10": ["G934", "R410", "R4182"],
        "icd9": ["3483", "78009", "7800"]
    },
    "transient_ischemic_attack": {
        "icd10": ["G45"],
        "icd9": ["435"]
    },
    "neuropathy_neurodegenerative": {
        "icd10": ["G20", "G30", "G35", "G50", "G51", "G60", "G61", "G62"],
        "icd9": ["332", "3310", "340", "350", "351", "356", "357"]
    }
}

NEURO_DISEASE_CLASSES = [
    "stroke_ischemic",
    "intracranial_hemorrhage",
    "epilepsy_seizures",
    "altered_mental_status_encephalopathy",
    "transient_ischemic_attack",
    "neuropathy_neurodegenerative"
]

NEURO_COMPLAINT_KEYWORDS = {
    "altered mental status": ["altered mental status", "ams", "confusion", "disoriented", "delirium", "unresponsive", "obtunded", "somnolent", "encephalopathy", "lethargic", "unconscious"],
    "seizure": ["seizure", "convulsion", "epilepsy", "postictal", "shaking", "clonic", "tonic", "ictal"],
    "focal weakness": ["weakness", "hemiparesis", "hemiplegia", "motor deficit", "loss of strength", "flaccid", "paralysis", "extremity weakness", "arm drift"],
    "headache": ["headache", "cephalea", "migraine", "head pain", "thunderclap", "occipital pain", "temporal pain"],
    "speech difficulty": ["speech", "aphasia", "dysarthria", "slurred speech", "expressive aphasia", "word finding", "dysphasia", "difficulty speaking"],
    "syncope/dizziness": ["syncope", "dizzy", "dizziness", "vertigo", "lightheaded", "passed out", "loss of consciousness", "loc", "presyncope", "near syncope"],
    "numbness/sensory": ["numbness", "paresthesia", "tingling", "sensory loss", "hypesthesia", "burning sensation", "pins and needles"],
    "gait instability": ["gait", "ataxia", "unsteady", "fall", "balance loss", "coordination", "stumbling", "off balance"],
    "visual disturbance": ["vision", "diplopia", "double vision", "visual field", "hemianopia", "blurriness", "blindness", "scotoma", "blurred vision"],
    "facial droop": ["facial droop", "facial asymmetry", "cranial nerve", "bells palsy", "facial weakness", "facial palsy", "mouth droop", "asymmetric smile"],
    "neurological assessment": ["neurology", "neuro consult", "neurological exam", "rule out stroke", "code stroke", "neurological evaluation"]
}

NEURO_COMPLAINT_CLASSES = list(NEURO_COMPLAINT_KEYWORDS.keys())
URGENCY_CLASSES = ["Emergency", "Urgent", "Routine"]

def clean_pre_diagnostic_narrative(text):
    if not text:
        return ""
    
    # 1. Truncate at the earliest occurrence of any post-diagnostic / inpatient / discharge section header
    split_pattern = re.compile(
        r'(?i)\b(?:brief hospital course|discharge diagnos(?:is|es)|final diagnos(?:is|es)|'
        r'discharge condition|discharge disposition|discharge instructions|'
        r'medications on discharge|primary diagnos(?:is|es)|secondary diagnos(?:is|es))\b'
    )
    
    m = split_pattern.search(text)
    if m:
        text = text[:m.start()]
        
    # 2. Scrub any residual isolated leak phrases
    scrub_patterns = [
        r'(?i)\bdischarge diagnosis\b',
        r'(?i)\bdischarge diagnoses\b',
        r'(?i)\bfinal diagnosis\b',
        r'(?i)\bfinal diagnoses\b',
        r'(?i)\bsecondary diagnosis\b',
        r'(?i)\bsecondary diagnoses\b',
        r'(?i)\bprincipal diagnosis\b',
        r'(?i)\bprimary diagnosis\b',
        r'(?i)\bbrief hospital course\b',
        r'(?i)\bdischarge condition\b',
        r'(?i)\bdischarge disposition\b'
    ]
    for sp in scrub_patterns:
        text = re.sub(sp, '', text)
        
    # 3. Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_neurology_complaint(text, ed_complaint=""):
    combined = f"{ed_complaint} {text}".lower()
    for complaint, kw_list in NEURO_COMPLAINT_KEYWORDS.items():
        for kw in kw_list:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined):
                return complaint
    return "neurological assessment"

def map_ed_acuity(acuity_str):
    try:
        ac = float(acuity_str)
        if ac in [1.0, 2.0]: return "Emergency"
        elif ac == 3.0: return "Urgent"
        else: return "Routine"
    except Exception:
        return "Urgent"

def build_neurology_dataset(diag_path="mimic-iv-3.1/hosp/diagnoses_icd.csv.gz",
                            note_path="note/discharge.csv.gz",
                            edstays_path="mimic-iv-ed-2.2/ed/edstays.csv.gz",
                            triage_path="mimic-iv-ed-2.2/ed/triage.csv.gz",
                            output_path="datasets/processed/neurologist/neurology_full_cohort.json"):
    print("=== STARTING NEUROLOGIST COHORT INGESTION & TEMPORAL LINKAGE ===")
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Scan ICD-9 & ICD-10 Diagnoses
    hadm_to_diseases = collections.defaultdict(set)
    hadm_to_subj = {}
    
    with gzip.open(diag_path, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm = row["hadm_id"]
            subj = row["subject_id"]
            code = row["icd_code"].strip().upper().replace(".", "")
            ver = row.get("icd_version", "10")
            
            for disease, patterns in NEUROLOGY_ICD_TAXONOMY.items():
                icd_list = patterns["icd10"] if ver == "10" else patterns["icd9"]
                for prefix in icd_list:
                    if code.startswith(prefix):
                        hadm_to_diseases[hadm].add(disease)
                        hadm_to_subj[hadm] = subj
                        break
                        
    neuro_hadm_ids = set(hadm_to_diseases.keys())
    print(f"[ICD Scanned] Found {len(neuro_hadm_ids):,} neurological encounters across {len(set(hadm_to_subj.values())):,} unique patients.")
    
    # 2. Link ED Stays and Triage Data temporally by exact HADM_ID
    hadm_to_ed_triage = {}
    if os.path.exists(edstays_path) and os.path.exists(triage_path):
        stay_to_hadm = {}
        with gzip.open(edstays_path, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stay_id = row["stay_id"]
                hadm_id = row.get("hadm_id", "")
                if hadm_id and hadm_id in neuro_hadm_ids:
                    stay_to_hadm[stay_id] = hadm_id
                    
        with gzip.open(triage_path, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stay_id = row["stay_id"]
                if stay_id in stay_to_hadm:
                    hadm = stay_to_hadm[stay_id]
                    hadm_to_ed_triage[hadm] = {
                        "chiefcomplaint": row.get("chiefcomplaint", ""),
                        "acuity": row.get("acuity", "3"),
                        "temperature": row.get("temperature", ""),
                        "heartrate": row.get("heartrate", ""),
                        "resprate": row.get("resprate", ""),
                        "o2sat": row.get("o2sat", ""),
                        "sbp": row.get("sbp", ""),
                        "dbp": row.get("dbp", ""),
                        "pain": row.get("pain", "")
                    }
    print(f"[ED Triage Linked] Loaded exact encounter ED triage info for {len(hadm_to_ed_triage):,} neurological admissions.")
    
    # 3. Scan Discharge Notes & Extract strictly Pre-Diagnostic Text
    valid_records = []
    hadm_seen_in_notes = set()
    
    with gzip.open(note_path, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm = row["hadm_id"]
            if hadm in neuro_hadm_ids and hadm not in hadm_seen_in_notes:
                hadm_seen_in_notes.add(hadm)
                subj = hadm_to_subj[hadm]
                raw_text = row["text"]
                
                clean_text = clean_pre_diagnostic_narrative(raw_text)
                if len(clean_text) < 50:
                    continue
                    
                ed_info = hadm_to_ed_triage.get(hadm, {})
                ed_comp = ed_info.get("chiefcomplaint", "")
                acuity = ed_info.get("acuity", "3")
                
                complaint_label = extract_neurology_complaint(clean_text, ed_comp)
                urgency_label = map_ed_acuity(acuity)
                disease_labels = sorted(list(hadm_to_diseases[hadm]))
                
                # Prepend ED triage vitals if available at presentation
                vitals_prefix = ""
                if ed_info.get("sbp") or ed_info.get("heartrate") or ed_info.get("temperature"):
                    vitals_prefix = (
                        f"Triage Vitals: BP={ed_info.get('sbp', 'NA')}/{ed_info.get('dbp', 'NA')} "
                        f"HR={ed_info.get('heartrate', 'NA')} RR={ed_info.get('resprate', 'NA')} "
                        f"SpO2={ed_info.get('o2sat', 'NA')}% Temp={ed_info.get('temperature', 'NA')}F "
                        f"Acuity={urgency_label}. "
                    )
                    
                full_input_text = vitals_prefix + clean_text
                
                valid_records.append({
                    "encounter_id": hadm,
                    "patient_id": subj,
                    "text": full_input_text,
                    "neurology_disease_labels": disease_labels,
                    "chief_complaint": complaint_label,
                    "urgency": urgency_label,
                    "ed_acuity": acuity
                })
                
    duration = time.time() - t0
    unique_patients = len(set(r["patient_id"] for r in valid_records))
    print(f"\n=== NEUROLOGY COHORT GENERATION COMPLETED ({duration:.1f}s) ===")
    print(f"Total Validated Encounters: {len(valid_records):,}")
    print(f"Total Unique Patients: {unique_patients:,}")
    
    # Disease prevalence in final cohort
    dis_counts = collections.defaultdict(int)
    for r in valid_records:
        for d in r["neurology_disease_labels"]:
            dis_counts[d] += 1
            
    print("\nDisease Support in Final Linked Cohort:")
    for d in NEURO_DISEASE_CLASSES:
        c = dis_counts[d]
        print(f"  - {d}: {c:,} encounters ({c/len(valid_records)*100:.2f}%)")
        
    print("\nChief Complaint Support in Final Linked Cohort:")
    comp_counts = collections.defaultdict(int)
    for r in valid_records:
        comp_counts[r["chief_complaint"]] += 1
    for c_name, c_cnt in sorted(comp_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {c_name}: {c_cnt:,} encounters ({c_cnt/len(valid_records)*100:.2f}%)")
        
    print("\nClinical Urgency Support in Final Linked Cohort:")
    urg_counts = collections.defaultdict(int)
    for r in valid_records:
        urg_counts[r["urgency"]] += 1
    for u_name in URGENCY_CLASSES:
        u_cnt = urg_counts[u_name]
        print(f"  - {u_name}: {u_cnt:,} encounters ({u_cnt/len(valid_records)*100:.2f}%)")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_records, f, indent=2)
    print(f"Saved verified cohort to {output_path}")
    return valid_records

if __name__ == "__main__":
    build_neurology_dataset()
