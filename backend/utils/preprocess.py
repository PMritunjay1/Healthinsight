import os
import sys
# Add project root to sys.path to enable backend module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# RULE ENGINE INTEGRATION
# ──────────────────────────────────────────────
from backend.utils.rule_engine import RuleEngine
rule_engine = RuleEngine()

def apply_synonym_mapping(text: str) -> str:
    return rule_engine.apply_synonym_mapping(text)

def map_domain(specialty: str, text: str) -> str:
    return rule_engine.map_domain(specialty, text)

def map_urgency(text: str) -> str:
    return rule_engine.map_urgency(text)

def map_task_type(specialty: str, text: str) -> str:
    return rule_engine.map_task_type(specialty, text)

# ──────────────────────────────────────────────
# PROCESSING PIPELINE
# ──────────────────────────────────────────────

def preprocess_intent_dataset():
    raw_path = 'datasets/raw/intent/mtsamples.csv'
    processed_dir = 'datasets/processed/intent'
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw Intent dataset not found at {raw_path}. Run downloader first.")
        
    # Load dataset
    df = pd.read_csv(raw_path)
    
    # 1. Clean missing values in transcription
    df = df.dropna(subset=['transcription'])
    
    # 1b. Drop duplicate transcriptions to prevent overfitting
    df = df.drop_duplicates(subset=['transcription'])
    
    # 2. Normalize and apply Synonym Mapping
    df['cleaned_text'] = df['transcription'].apply(apply_synonym_mapping)
    
    # 3. Apply Multi-Task Target Mapping
    df['medical_domain'] = df.apply(lambda r: map_domain(r['medical_specialty'], r['cleaned_text']), axis=1)
    df['urgency_level'] = df['cleaned_text'].apply(map_urgency)
    df['task_type'] = df.apply(lambda r: map_task_type(r['medical_specialty'], r['cleaned_text']), axis=1)
    
    # 4. Filter columns and perform split
    processed_df = df[['cleaned_text', 'medical_domain', 'urgency_level', 'task_type']]
    
    train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['medical_domain'])
    
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Preprocessed Intent dataset:")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Validation samples: {len(val_df)}")
    print(f" - Outputs saved to {processed_dir}")
    
    return train_df, val_df

def preprocess_diagnosis_dataset():
    raw_path = 'datasets/raw/diagnosis/Symptom2Disease.csv'
    processed_dir = 'datasets/processed/diagnosis'
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw Diagnosis dataset not found at {raw_path}. Run downloader first.")
        
    df = pd.read_csv(raw_path)
    
    # 1. Clean missing values in text
    df = df.dropna(subset=['text'])
    
    # 1b. Drop duplicate symptom texts to prevent overfitting
    df = df.drop_duplicates(subset=['text'])
    
    # 2. Normalize and apply Synonym Mapping
    df['cleaned_text'] = df['text'].apply(apply_synonym_mapping)
    
    # 3. Target label is 'label'
    processed_df = df[['cleaned_text', 'label']]
    
    train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['label'])
    
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Preprocessed Diagnosis dataset:")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Validation samples: {len(val_df)}")
    print(f" - Outputs saved to {processed_dir}")
    
    return train_df, val_df

def preprocess_risk_dataset():
    raw_path = 'datasets/raw/risk/synthetic_medical_triage.csv'
    processed_dir = 'datasets/processed/risk'
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw Risk dataset not found at {raw_path}. Run downloader first.")
        
    df = pd.read_csv(raw_path)
    
    # Generate synthesized text clinical notes from structured columns
    def synthesize_triage_note(row) -> str:
        age_str = f"{row['age']:.1f}" if not pd.isna(row['age']) else "unknown"
        hr = f"{row['heart_rate']:.1f}" if not pd.isna(row['heart_rate']) else "unknown"
        bp = f"{row['systolic_blood_pressure']:.1f}" if not pd.isna(row['systolic_blood_pressure']) else "unknown"
        o2 = f"{row['oxygen_saturation']:.1f}" if not pd.isna(row['oxygen_saturation']) else "unknown"
        temp = f"{row['body_temperature']:.1f}" if not pd.isna(row['body_temperature']) else "unknown"
        pain = f"{int(row['pain_level'])}" if not pd.isna(row['pain_level']) else "unknown"
        chronics = f"{int(row['chronic_disease_count'])}" if not pd.isna(row['chronic_disease_count']) else "0"
        visits = f"{int(row['previous_er_visits'])}" if not pd.isna(row['previous_er_visits']) else "0"
        arrival = str(row['arrival_mode']).lower().replace('_', ' ')
        
        note = (
            f"patient is a {age_str}-year-old who presented as a {arrival}. "
            f"chief complaint is severe discomfort with a pain score of {pain}/10. "
            f"vitals on admission: pulse rate {hr} bpm, systolic blood pressure {bp} mmhg, "
            f"oxygen saturation {o2}%, and body temperature {temp} c. "
            f"past medical history is significant for {chronics} chronic conditions and {visits} previous emergency department visits."
        )
        return note

    df['text'] = df.apply(synthesize_triage_note, axis=1)
    
    # 1. Clean missing values
    df = df.dropna(subset=['text'])
    
    # 1b. Drop duplicate texts
    df = df.drop_duplicates(subset=['text'])
    
    # 2. Normalize and apply Synonym Mapping
    df['cleaned_text'] = df['text'].apply(apply_synonym_mapping)
    
    # 3. Target label is 'triage_level'
    processed_df = df[['cleaned_text', 'triage_level']]
    
    train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df['triage_level'])
    
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Preprocessed Risk dataset:")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Validation samples: {len(val_df)}")
    print(f" - Outputs saved to {processed_dir}")
    
    return train_df, val_df

if __name__ == "__main__":
    preprocess_intent_dataset()
    preprocess_diagnosis_dataset()
    preprocess_risk_dataset()
