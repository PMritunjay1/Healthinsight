import os
import json
import random
import numpy as np

def split_cohort_patient_level(input_cohort_path, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_cohort_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    patient_to_records = {}
    for r in records:
        pid = r["patient_id"]
        if pid not in patient_to_records:
            patient_to_records[pid] = []
        patient_to_records[pid].append(r)
        
    unique_patients = list(patient_to_records.keys())
    random.shuffle(unique_patients)
    
    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train+n_val])
    test_patients = set(unique_patients[n_train+n_val:])
    
    train_records = [r for pid in train_patients for r in patient_to_records[pid]]
    val_records = [r for pid in val_patients for r in patient_to_records[pid]]
    test_records = [r for pid in test_patients for r in patient_to_records[pid]]
    
    with open(os.path.join(output_dir, "train_split.json"), "w", encoding="utf-8") as f: json.dump(train_records, f, indent=2)
    with open(os.path.join(output_dir, "val_split.json"), "w", encoding="utf-8") as f: json.dump(val_records, f, indent=2)
    with open(os.path.join(output_dir, "test_split.json"), "w", encoding="utf-8") as f: json.dump(test_records, f, indent=2)
    
    print(f"Split Summary: Train={len(train_records)} ({len(train_patients)} pts), Val={len(val_records)} ({len(val_patients)} pts), Test={len(test_records)} ({len(test_patients)} pts)")

if __name__ == "__main__":
    print("split_dataset module ready.")
