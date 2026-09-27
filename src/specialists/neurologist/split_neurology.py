import os
import sys
import io
import json
import random
import collections
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def split_neurology_cohort(input_path="datasets/processed/neurologist/neurology_full_cohort.json",
                           output_dir="datasets/processed/neurologist",
                           train_ratio=0.70,
                           val_ratio=0.15,
                           test_ratio=0.15,
                           seed=42):
    print("=== EXECUTING STRICT PATIENT-LEVEL SPLITTING FOR NEUROLOGIST ===")
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    patient_to_records = collections.defaultdict(list)
    for r in records:
        pid = r["patient_id"]
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
    
    # Assertions
    ov_tv = len(train_patients.intersection(val_patients))
    ov_tt = len(train_patients.intersection(test_patients))
    ov_vt = len(val_patients.intersection(test_patients))
    
    print(f"Total Cohort: {len(records):,} records across {n_total:,} unique patients.")
    print(f"  - Train Split: {len(train_records):,} records ({len(train_patients):,} patients, {len(train_records)/len(records)*100:.2f}%)")
    print(f"  - Val Split:   {len(val_records):,} records ({len(val_patients):,} patients, {len(val_records)/len(records)*100:.2f}%)")
    print(f"  - Test Split:  {len(test_records):,} records ({len(test_patients):,} patients, {len(test_records)/len(records)*100:.2f}%)")
    
    print(f"Patient Overlap: Train_Val={ov_tv}, Train_Test={ov_tt}, Val_Test={ov_vt}")
    assert ov_tv == 0 and ov_tt == 0 and ov_vt == 0, "FATAL: Patient overlap detected!"
    
    # Duplicate encounter ID check
    all_hadms = [r["encounter_id"] for r in records]
    assert len(all_hadms) == len(set(all_hadms)), "FATAL: Duplicate encounter IDs in dataset!"
    
    # Save split files
    train_out = os.path.join(output_dir, "train_split.json")
    val_out = os.path.join(output_dir, "val_split.json")
    test_out = os.path.join(output_dir, "test_split.json")
    
    with open(train_out, "w", encoding="utf-8") as f: json.dump(train_records, f, indent=2)
    with open(val_out, "w", encoding="utf-8") as f: json.dump(val_records, f, indent=2)
    with open(test_out, "w", encoding="utf-8") as f: json.dump(test_records, f, indent=2)
    
    print(f"Saved splits to {output_dir}/")
    return len(train_records), len(val_records), len(test_records)

if __name__ == "__main__":
    split_neurology_cohort()
