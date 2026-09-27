import os
import sys
import gzip
import csv
import re
import json
import collections
import time

PULMONARY_ICD_TAXONOMY = {
    "asthma": {
        "icd10": ["J45"],
        "icd9": ["493"]
    },
    "copd": {
        "icd10": ["J44", "J41", "J42", "J43"],
        "icd9": ["491", "492", "496"]
    },
    "pneumonia": {
        "icd10": ["J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69"],
        "icd9": ["480", "481", "482", "483", "484", "485", "486", "507"]
    },
    "respiratory_failure": {
        "icd10": ["J96", "J80"],
        "icd9": ["51881", "51882", "51884", "5185"]
    },
    "pleural_effusion": {
        "icd10": ["J90", "J91"],
        "icd9": ["5119", "5118"]
    },
    "pulmonary_embolism": {
        "icd10": ["I26"],
        "icd9": ["4151"]
    }
}

PULM_DISEASE_CLASSES = ["asthma", "copd", "pneumonia", "respiratory_failure", "pleural_effusion", "pulmonary_embolism"]

PULM_COMPLAINT_KEYWORDS = {
    "dyspnea": ["dyspnea", "shortness of breath", "sob", "breathlessness", "difficulty breathing", "orthopnea", "air hunger"],
    "cough": ["cough", "productive cough", "dry cough", "paroxysmal cough", "barking cough"],
    "wheezing": ["wheeze", "wheezing", "stridor", "airway constriction", "bronchospasm"],
    "chest pain": ["chest pain", "pleuritic", "pleurisy", "substernal", "chest tightness", "chest pressure"],
    "fever": ["fever", "chills", "febrile", "pyrexia", "night sweats"],
    "hemoptysis": ["hemoptysis", "coughing blood", "blood in sputum", "blood tinged sputum"],
    "sputum production": ["sputum", "phlegm", "mucus", "purulent sputum", "yellow sputum", "green sputum"],
    "fatigue": ["fatigue", "weakness", "lethargy", "exhaustion", "malaise"],
    "altered mental status": ["altered mental status", "ams", "confusion", "somnolence", "hypercapnia", "drowsy", "lethargic"],
    "syncope": ["syncope", "passed out", "blackout", "dizziness", "lightheaded", "fainted"],
    "respiratory assessment": ["respiratory", "pulmonary consult", "hypoxia", "low oxygen", "desaturation"]
}

def clean_pre_diagnostic_narrative(text):
    if not text:
        return ""
    # Strip post-diagnostic sections
    text = re.sub(r'(?i)Brief Hospital Course:[\s\S]*?(?=(Discharge Disposition|Discharge Diagnosis|Medications on Discharge|$))', '', text)
    text = re.sub(r'(?i)Discharge Diagnosis:[\s\S]*?(?=(Discharge Condition|Discharge Instructions|$))', '', text)
    text = re.sub(r'(?i)Discharge Disposition:.*', '', text)
    text = re.sub(r'(?i)Discharge Condition:.*', '', text)
    text = re.sub(r'(?i)Secondary Diagnosis:.*', '', text)
    text = re.sub(r'(?i)Final Diagnosis:.*', '', text)
    
    # Prune extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pulmonary_complaint(text, ed_complaint=""):
    combined = f"{ed_complaint} {text}".lower()
    for complaint, kw_list in PULM_COMPLAINT_KEYWORDS.items():
        for kw in kw_list:
            if re.search(r'\b' + re.escape(kw) + r'\b', combined):
                return complaint
    return "respiratory assessment"

def map_ed_acuity(acuity_str):
    try:
        ac = float(acuity_str)
        if ac in [1.0, 2.0]: return "Emergency"
        elif ac == 3.0: return "Urgent"
        else: return "Routine"
    except Exception:
        return "Urgent"

def build_pulmonary_dataset(diag_path="mimic-iv-3.1/hosp/diagnoses_icd.csv.gz",
                            note_path="note/discharge.csv.gz",
                            triage_path="mimic-iv-ed-2.2/ed/triage.csv.gz",
                            output_path="datasets/processed/pulmonologist/pulmonary_full_cohort.json"):
    print("=== STARTING PULMONOLOGIST COHORT INGESTION & LINKAGE ===")
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
            
            for disease, patterns in PULMONARY_ICD_TAXONOMY.items():
                icd_list = patterns["icd10"] if ver == "10" else patterns["icd9"]
                for prefix in icd_list:
                    if code.startswith(prefix):
                        hadm_to_diseases[hadm].add(disease)
                        hadm_to_subj[hadm] = subj
                        break
                        
    pulm_hadm_ids = set(hadm_to_diseases.keys())
    print(f"[ICD Scanned] Found {len(pulm_hadm_ids)} pulmonary encounters across {len(set(hadm_to_subj.values()))} unique patients.")
    
    # 2. Scan ED Triage Data
    subj_to_ed_info = {}
    if os.path.exists(triage_path):
        with gzip.open(triage_path, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row["subject_id"]
                if subj in hadm_to_subj.values():
                    subj_to_ed_info[subj] = {
                        "ed_complaint": row.get("chiefcomplaint", ""),
                        "acuity": row.get("acuity", "3"),
                        "temp": row.get("temperature", ""),
                        "hr": row.get("heartrate", ""),
                        "rr": row.get("resprate", ""),
                        "spo2": row.get("o2sat", ""),
                        "sbp": row.get("sbp", ""),
                        "dbp": row.get("dbp", "")
                    }
    print(f"[ED Triage Linked] Loaded ED triage info for {len(subj_to_ed_info)} pulmonary patients.")
    
    # 3. Scan Discharge Notes & Extract Pre-Diagnostic Text
    valid_records = []
    hadm_seen_in_notes = set()
    
    with gzip.open(note_path, "rt", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm = row["hadm_id"]
            if hadm in pulm_hadm_ids and hadm not in hadm_seen_in_notes:
                hadm_seen_in_notes.add(hadm)
                subj = hadm_to_subj[hadm]
                raw_text = row["text"]
                
                clean_text = clean_pre_diagnostic_narrative(raw_text)
                if len(clean_text) < 50:
                    continue
                    
                ed_info = subj_to_ed_info.get(subj, {})
                ed_comp = ed_info.get("ed_complaint", "")
                acuity = ed_info.get("acuity", "3")
                
                complaint_label = extract_pulmonary_complaint(clean_text, ed_comp)
                urgency_label = map_ed_acuity(acuity)
                disease_labels = sorted(list(hadm_to_diseases[hadm]))
                
                # Prepend triage vitals if available
                vitals_prefix = ""
                if ed_info.get("spo2") or ed_info.get("rr"):
                    vitals_prefix = f"Triage Vitals: SpO2={ed_info.get('spo2', 'NA')}% RR={ed_info.get('rr', 'NA')} HR={ed_info.get('hr', 'NA')} Temp={ed_info.get('temp', 'NA')}F BP={ed_info.get('sbp', 'NA')}/{ed_info.get('dbp', 'NA')}. "
                
                full_input_text = vitals_prefix + clean_text
                
                valid_records.append({
                    "encounter_id": hadm,
                    "patient_id": subj,
                    "text": full_input_text,
                    "pulmonary_disease_labels": disease_labels,
                    "chief_complaint": complaint_label,
                    "urgency": urgency_label,
                    "ed_acuity": acuity
                })
                
    duration = time.time() - t0
    unique_patients = len(set(r["patient_id"] for r in valid_records))
    print(f"\n=== PULMONARY COHORT GENERATION COMPLETED ({duration:.1f}s) ===")
    print(f"Total Validated Encounters: {len(valid_records)}")
    print(f"Total Unique Patients: {unique_patients}")
    
    # Disease prevalence in final cohort
    dis_counts = collections.defaultdict(int)
    for r in valid_records:
        for d in r["pulmonary_disease_labels"]:
            dis_counts[d] += 1
            
    print("\nDisease Support in Final Linked Cohort:")
    for d, c in sorted(dis_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {d}: {c} encounters ({c/len(valid_records)*100:.2f}%)")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_records, f, indent=2)
    print(f"Saved verified cohort to {output_path}")
    return valid_records

if __name__ == "__main__":
    build_pulmonary_dataset()
