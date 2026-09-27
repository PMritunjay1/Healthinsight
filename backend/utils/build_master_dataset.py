import os
import re
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# CONFIGURATION & DICTIONARIES
# ──────────────────────────────────────────────

ABBREVIATIONS = {
    "sob": "shortness of breath",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "uri": "upper respiratory infection",
    "uti": "urinary tract infection",
    "gerd": "gastroesophageal reflux disease",
    "pud": "peptic ulcer disease",
    "bp": "blood pressure",
    "hr": "heart rate"
}

SYNONYMS = {
    "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing", "sob"],
    "fever": ["pyrexia", "high temperature", "febrile", "chills"],
    "chest pain": ["angina", "chest discomfort", "chest pressure", "chest tightness"],
    "headache": ["cephalalgia", "migraine headache", "throbbing head pain"],
    "nausea": ["vomiting", "queasiness", "nauseous feeling"],
    "swelling": ["edema", "swollen tissues", "fluid retention"],
    "palpitations": ["racing heart", "heart pounding", "fluttering chest"],
    "fatigue": ["tiredness", "exhaustion", "lethargy", "weakness"],
    "hypertension": ["high blood pressure", "elevated bp", "htn"],
    "diabetes": ["diabetes mellitus", "high blood sugar", "diabetic condition"]
}

COMMON_TYPOS = {
    "shortness of breath": "shorness of breath",
    "difficulty breathing": "difficutly breathing",
    "hypertension": "hypertention",
    "pneumonia": "pneumona",
    "migraine": "migrane",
    "nausea": "nauseea",
    "palpitations": "palpatations",
    "diarrhea": "diarhea"
}

TEMPLATE_FRAMES = [
    "Patient reports: {text}",
    "Clinical notes state: {text}",
    "Subject complains of: {text}",
    "Admitted with following symptoms: {text}",
    "Narrative: {text}",
    "{text}"
]

# ──────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # remove double spacing
    text = text.strip()
    return text

def normalize_disease_name(disease: str) -> str:
    disease = disease.lower().strip()
    # Normalize common variations
    if "myocardial" in disease or "heart attack" in disease:
        return "myocardial infarction"
    if "diabetes" in disease:
        return "diabetes mellitus"
    if "asthma" in disease:
        return "bronchial asthma"
    if "gerd" in disease or "gastroesophageal" in disease:
        return "gastroesophageal reflux disease"
    if "ulcer" in disease:
        return "peptic ulcer disease"
    return disease

def expand_abbreviations(text: str) -> str:
    text_lower = text.lower()
    for abbr, full in ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text_lower = re.sub(pattern, full, text_lower)
    return text_lower

def augment_sample(text: str, disease: str, is_minor: bool = False) -> list:
    """
    Generates deterministic augmented versions of the input text:
    1. Synonym Replacement
    2. Abbreviation Contraction
    3. Spelling Typos introduction
    4. Template Framing (all templates if is_minor is True, else just one)
    """
    augmented = []
    
    # 1. Synonym Replacement
    syn_texts = [text]
    for term, syn_list in SYNONYMS.items():
        if term in text:
            for syn in syn_list:
                syn_texts.append(text.replace(term, syn))
            if not is_minor:
                # Only keep the first replacement for major classes
                syn_texts = [text, text.replace(term, syn_list[0])]
                break
    augmented.extend(syn_texts)

    # 2. Abbreviation Contraction
    abbr_texts = []
    for syn_t in syn_texts:
        for abbr, full in ABBREVIATIONS.items():
            if full in syn_t:
                abbr_texts.append(syn_t.replace(full, abbr.upper()))
                if not is_minor:
                    break
    augmented.extend(abbr_texts)

    # 3. Spelling Typos
    typo_texts = []
    current_pool = list(set(augmented + [text]))
    for t_val in current_pool:
        for correct, typo in COMMON_TYPOS.items():
            if correct in t_val:
                typo_texts.append(t_val.replace(correct, typo))
                if not is_minor:
                    break
    augmented.extend(typo_texts)

    # 4. Template Framing
    final_pool = list(set(augmented + [text]))
    framed_texts = []
    for t_val in final_pool:
        if is_minor:
            # Apply all templates for minor classes to balance sample size
            for template in TEMPLATE_FRAMES:
                framed_texts.append(template.format(text=t_val))
        else:
            # Apply just one template deterministically for major classes
            frame_idx = (len(t_val) + len(disease)) % len(TEMPLATE_FRAMES)
            framed_texts.append(TEMPLATE_FRAMES[frame_idx].format(text=t_val))
            
    augmented.extend(framed_texts)
    
    # Clean up and return unique versions (excluding original text)
    result = list(set(augmented))
    if text in result:
        result.remove(text)
    return result



# ──────────────────────────────────────────────
# MASTER DATASET GENERATOR
# ──────────────────────────────────────────────

def build_master_dataset():
    print("==================================================")
    print("GENERATING UNIFIED MASTER MEDICAL DATASET")
    print("==================================================")
    
    raw_records = []
    
    # 1. Ingest Symptom2Disease
    s2d_path = 'datasets/raw/diagnosis/Symptom2Disease.csv'
    if os.path.exists(s2d_path):
        df_s2d = pd.read_csv(s2d_path)
        print(f"Loaded Symptom2Disease: {len(df_s2d)} raw rows.")
        for _, row in df_s2d.iterrows():
            text = str(row['text'])
            disease = normalize_disease_name(str(row['label']))
            
            # Map Specialty
            if disease in ["hypertension", "angina"]:
                specialty = "Cardiologist"
            elif disease in ["bronchial asthma", "pneumonia"]:
                specialty = "Pulmonologist"
            elif disease in ["migraine", "cervical spondylosis"]:
                specialty = "Neurologist"
            else:
                specialty = "General"
                
            raw_records.append({
                "Patient Text": text,
                "Disease": disease,
                "Specialty": specialty,
                "Source Dataset": "Symptom2Disease"
            })
            
    # 2. Ingest MTSamples
    mt_path = 'datasets/raw/intent/mtsamples.csv'
    if os.path.exists(mt_path):
        df_mt = pd.read_csv(mt_path)
        print(f"Loaded MTSamples: {len(df_mt)} raw rows.")
        df_mt = df_mt.dropna(subset=['transcription', 'medical_specialty'])
        for _, row in df_mt.iterrows():
            text = str(row['transcription'])
            spec_raw = str(row['medical_specialty']).lower().strip()
            desc = str(row['description']).lower()
            
            # Map Specialty and Disease based on raw category
            specialty = "General"
            disease = "unspecified"
            
            if "cardiovascular" in spec_raw:
                specialty = "Cardiologist"
                disease = "hypertension" if "hypertension" in desc else "angina"
            elif "pulmonary" in spec_raw:
                specialty = "Pulmonologist"
                disease = "bronchial asthma" if "asthma" in desc else "pneumonia"
            elif "neurology" in spec_raw or "neurosurgery" in spec_raw:
                specialty = "Neurologist"
                disease = "migraine" if "headache" in desc else "cervical spondylosis"
            elif "gastroenterology" in spec_raw:
                specialty = "General"
                disease = "gastroesophageal reflux disease"
                
            raw_records.append({
                "Patient Text": text,
                "Disease": disease,
                "Specialty": specialty,
                "Source Dataset": "MTSamples"
            })
            
    # 3. Ingest Triage (NHAMCS Triage)
    triage_path = 'datasets/raw/risk/synthetic_medical_triage.csv'
    if os.path.exists(triage_path):
        df_triage = pd.read_csv(triage_path)
        print(f"Loaded Triage: {len(df_triage)} raw rows.")
        
        # Filter a subset to enrich the Cardiologist/Pulmonologist categories
        for idx, row in df_triage.head(1500).iterrows():
            pain = int(row['pain_level']) if not pd.isna(row['pain_level']) else 0
            # Construct a patient narrative
            arrival = str(row['arrival_mode']).lower().replace('_', ' ')
            hr = int(row['heart_rate']) if not pd.isna(row['heart_rate']) else 72
            bp = f"{int(row['systolic_blood_pressure'])}/90" if not pd.isna(row['systolic_blood_pressure']) else "120/80"
            
            if pain > 7 and hr > 100:
                text = f"Patient arrived via {arrival} complaining of sudden severe chest pressure and palpitations. Heart rate is {hr} bpm, BP is {bp}."
                specialty = "Cardiologist"
                disease = "angina"
            elif hr > 90:
                text = f"Patient describes shortness of breath and wheezing at night. Onset was gradual. BP is {bp}, pulse rate is {hr} bpm."
                specialty = "Pulmonologist"
                disease = "bronchial asthma"
            else:
                text = f"Patient complains of severe throbbing headache, neck pain and mild nausea. Vital signs: pulse {hr} bpm, BP {bp}."
                specialty = "Neurologist"
                disease = "migraine"
                
            raw_records.append({
                "Patient Text": text,
                "Disease": disease,
                "Specialty": specialty,
                "Source Dataset": "Triage"
            })

    master_df = pd.DataFrame(raw_records)
    print(f"Aggregated raw master pool: {len(master_df)} rows.")

    # ──────────────────────────────────────────────
    # CLEANING PIPELINE
    # ──────────────────────────────────────────────
    
    # 1. Clean whitespace and lowercase
    master_df["Patient Text"] = master_df["Patient Text"].apply(clean_text)
    
    # 2. Drop rows with empty text
    master_df = master_df[master_df["Patient Text"] != ""]
    
    # 3. Deduplicate
    initial_count = len(master_df)
    master_df = master_df.drop_duplicates(subset=["Patient Text"])
    duplicate_count = initial_count - len(master_df)
    print(f"Deduplicated: removed {duplicate_count} duplicate clinical records.")

    # 4. Standardize / expand abbreviations
    master_df["Patient Text"] = master_df["Patient Text"].apply(expand_abbreviations)

    # ──────────────────────────────────────────────
    # TRAIN / VALIDATION / TEST SPLIT (NO AUGMENTATION YET)
    # ──────────────────────────────────────────────
    
    # Perform stratified split to ensure representation of all Specialties across splits
    train_val_df, test_df = train_test_split(
        master_df, 
        test_size=0.10, 
        random_state=42, 
        stratify=master_df["Specialty"]
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.1111,  # 0.1111 * 0.90 ≈ 0.10 of total
        random_state=42, 
        stratify=train_val_df["Specialty"]
    )
    
    # Set explicit splits
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["Split"] = "Train"
    val_df["Split"] = "Validation"
    test_df["Split"] = "Test"
    
    # ──────────────────────────────────────────────
    # AUGMENTATION PIPELINE (TRAIN ONLY)
    # ──────────────────────────────────────────────
    print("Performing controlled data augmentation on Train split...")
    
    augmented_records = []
    
    for _, row in train_df.iterrows():
        text = row["Patient Text"]
        disease = row["Disease"]
        specialty = row["Specialty"]
        source = row["Source Dataset"]
        
        # Generate augmented versions (targeted balancing for minor classes)
        is_minor = specialty in ["Cardiologist", "Pulmonologist"]
        aug_versions = augment_sample(text, disease, is_minor=is_minor)
        for aug_text in aug_versions:
            augmented_records.append({
                "Patient Text": aug_text,
                "Disease": disease,
                "Specialty": specialty,
                "Source Dataset": source,
                "Split": "Train"
            })
            
    df_aug = pd.DataFrame(augmented_records)
    print(f"Generated {len(df_aug)} augmented training samples.")
    
    # Combine original train and augmented train
    augmented_train_df = pd.concat([train_df, df_aug], ignore_index=True)
    
    # Final master unified dataset
    final_master_df = pd.concat([augmented_train_df, val_df, test_df], ignore_index=True)
    
    # ──────────────────────────────────────────────
    # OUTPUT EXPORTS
    # ──────────────────────────────────────────────
    processed_dir = 'datasets/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save unified master dataset
    master_path = os.path.join(processed_dir, 'master_medical_dataset.csv')
    final_master_df.to_csv(master_path, index=False)
    print(f"Saved master medical dataset to {master_path} (Total samples: {len(final_master_df)})")
    
    # Auto-generate specialty datasets
    specs = ["Cardiologist", "Pulmonologist", "Neurologist", "General"]
    specialty_stats = {}
    
    for spec in specs:
        spec_df = final_master_df[final_master_df["Specialty"] == spec]
        spec_dir = os.path.join(processed_dir, spec.lower())
        os.makedirs(spec_dir, exist_ok=True)
        
        # Save split subsets for training
        train_sub = spec_df[spec_df["Split"] == "Train"]
        val_sub = spec_df[spec_df["Split"] == "Validation"]
        test_sub = spec_df[spec_df["Split"] == "Test"]
        
        train_sub.to_csv(os.path.join(spec_dir, 'train.csv'), index=False)
        val_sub.to_csv(os.path.join(spec_dir, 'val.csv'), index=False)
        test_sub.to_csv(os.path.join(spec_dir, 'test.csv'), index=False)
        
        specialty_stats[spec] = {
            "Total": len(spec_df),
            "Train": len(train_sub),
            "Validation": len(val_sub),
            "Test": len(test_sub),
            "Diseases": list(spec_df["Disease"].unique())
        }
        
    # Generate final stats dictionary for reporting
    report_dict = {
        "total_samples": len(final_master_df),
        "deduplicated_count": duplicate_count,
        "splits": {
            "Train": len(final_master_df[final_master_df["Split"] == "Train"]),
            "Validation": len(final_master_df[final_master_df["Split"] == "Validation"]),
            "Test": len(final_master_df[final_master_df["Split"] == "Test"])
        },
        "source_contributions": final_master_df["Source Dataset"].value_counts().to_dict(),
        "specialties": specialty_stats,
        "augmentation_stats": {
            "original_train_count": len(train_df),
            "augmented_train_count": len(df_aug)
        }
    }
    
    return report_dict

if __name__ == "__main__":
    stats = build_master_dataset()
    print("\nMaster Dataset Build Complete. Summary:")
    print(stats)
