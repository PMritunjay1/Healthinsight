import os
import sys
import json
import gzip
import csv
import re

DISEASE_ICD_MAP = {
    "hypertension": ["I10", "I11", "I12", "I13", "I15", "401", "402", "403", "404", "405"],
    "atrial fibrillation": ["I48", "42731", "42732"],
    "coronary artery disease": ["I25", "414", "410", "411", "412"],
    "heart failure": ["I50", "428"],
    "arrhythmia": ["I47", "I49", "427"],
    "angina": ["I20", "413"]
}

COMPLAINT_KEYWORDS = {
    "chest pain": ["chest pain", "cp", "angina", "substernal", "pressure in chest", "tightness"],
    "dyspnea": ["shortness of breath", "sob", "dyspnea", "breathlessness", "difficulty breathing", "orthopnea"],
    "syncope": ["syncope", "passed out", "blackout", "dizziness", "lightheaded", "presyncope", "fainted", "fall"],
    "palpitations": ["palpitations", "racing heart", "fluttering", "irregular heartbeat", "tachycardia"],
    "edema": ["edema", "swelling", "fluid retention", "leg swelling", "ankle swelling", "ascites"],
    "fatigue": ["fatigue", "weakness", "lethargy", "tiredness", "generalized weakness", "malaise"],
    "nausea": ["nausea", "vomiting", "epigastric", "indigestion", "abdominal pain", "gi upset"],
    "cough": ["cough", "productive cough", "hemoptysis", "wheezing"],
    "headache": ["headache", "occipital headache", "visual changes", "head pressure"],
    "hypertensive symptoms": ["hypertension", "elevated bp", "high blood pressure", "crisis"]
}

def map_icd_to_diseases(icd_codes):
    matched = set()
    for code in icd_codes:
        clean = code.replace(".", "").upper()
        for disease, prefixes in DISEASE_ICD_MAP.items():
            for prefix in prefixes:
                if clean.startswith(prefix):
                    matched.add(disease)
                    break
    if not matched:
        matched.add("hypertension")
    return sorted(list(matched))

def map_text_to_complaint(text):
    text_lower = text.lower()
    for complaint, keywords in COMPLAINT_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return complaint
    return "cardiac assessment"

def map_acuity_to_urgency(acuity_val):
    try:
        ac = float(acuity_val)
        if ac in [1.0, 2.0]:
            return "Emergency"
        elif ac == 3.0:
            return "Urgent"
        else:
            return "Routine"
    except Exception:
        return "Urgent"

def clean_clinical_narrative(text):
    # Strictly extract pre-diagnostic narrative and prune post-diagnostic sections
    text = re.sub(r'(?i)Brief Hospital Course:[\s\S]*?(?=(Discharge Disposition|Discharge Diagnosis|Medications on Discharge|$))', '', text)
    text = re.sub(r'(?i)Discharge Diagnosis:[\s\S]*?(?=(Discharge Condition|Discharge Instructions|$))', '', text)
    text = re.sub(r'(?i)Discharge Disposition:.*', '', text)
    text = re.sub(r'(?i)Discharge Condition:.*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Ingest module defined successfully.")
