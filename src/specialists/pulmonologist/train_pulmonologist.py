import os
import sys
import io
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, hamming_loss, accuracy_score,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def log_step(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {message}")
    sys.stdout.flush()

PULM_DISEASE_CLASSES = ["asthma", "copd", "pneumonia", "respiratory_failure", "pleural_effusion", "pulmonary_embolism"]
PULM_COMPLAINT_CLASSES = [
    "dyspnea", "cough", "wheezing", "chest pain", "fever", "hemoptysis",
    "sputum production", "fatigue", "altered mental status", "syncope", "respiratory assessment"
]
URGENCY_CLASSES = ["Emergency", "Urgent", "Routine"]

DISEASE_MAP = {lbl: idx for idx, lbl in enumerate(PULM_DISEASE_CLASSES)}
COMPLAINT_MAP = {lbl: idx for idx, lbl in enumerate(PULM_COMPLAINT_CLASSES)}
URGENCY_MAP = {lbl: idx for idx, lbl in enumerate(URGENCY_CLASSES)}

# ──────────────────────────────────────────────
# 1. DATASET CLASS
# ──────────────────────────────────────────────
class PulmonologistDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=384, desc="Dataset"):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = [r.get("text", "No clinical narrative.") for r in records]
        
        self.disease_targets = []
        for r in records:
            vec = np.zeros(len(PULM_DISEASE_CLASSES), dtype=np.float32)
            lbls = r.get("pulmonary_disease_labels", [])
            for d in lbls:
                if d in DISEASE_MAP:
                    vec[DISEASE_MAP[d]] = 1.0
            if vec.sum() == 0:
                vec[DISEASE_MAP["pneumonia"]] = 1.0
            self.disease_targets.append(vec)
            
        self.complaint_targets = [COMPLAINT_MAP.get(r.get("chief_complaint", "respiratory assessment"), 10) for r in records]
        self.urgency_targets = [URGENCY_MAP.get(r.get("urgency", "Emergency"), 0) for r in records]
        
        log_step(f"Tokenizing {desc} ({len(self.records)} samples, max_len={max_len})...")
        self.encodings = []
        for text in self.texts:
            enc = tokenizer(
                str(text),
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten()
            })
        log_step(f"Tokenization of {desc} complete.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.encodings[idx].copy()
        item['disease_labels'] = torch.tensor(self.disease_targets[idx], dtype=torch.float)
        item['complaint_label'] = torch.tensor(self.complaint_targets[idx], dtype=torch.long)
        item['urgency_label'] = torch.tensor(self.urgency_targets[idx], dtype=torch.long)
        return item

# ──────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ──────────────────────────────────────────────
class PulmonologistSpecialistModel(nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_diseases=6, num_complaints=11, num_urgencies=3):
        super().__init__()
        log_step(f"Initializing PubMedBERT backbone: {model_name}...")
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        self.fc_dis1 = nn.Linear(hidden_size, 256)
        self.fc_dis2 = nn.Linear(256, num_diseases)
        
        self.fc_comp1 = nn.Linear(hidden_size, 256)
        self.fc_comp2 = nn.Linear(256, num_complaints)
        
        self.fc_urg1 = nn.Linear(hidden_size, 256)
        self.fc_urg2 = nn.Linear(256, num_urgencies)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        rep = outputs[0][:, 0, :]
        
        d = self.dropout(self.relu(self.fc_dis1(rep)))
        disease_logits = self.fc_dis2(d)
        
        c = self.dropout(self.relu(self.fc_comp1(rep)))
        complaint_logits = self.fc_comp2(c)
        
        u = self.dropout(self.relu(self.fc_urg1(rep)))
        urgency_logits = self.fc_urg2(u)
        
        return disease_logits, complaint_logits, urgency_logits

# ONNX Export Wrapper
class PulmonologistSpecialistONNX(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.transformer = trained_model.transformer
        
        self.w_dis1 = nn.Parameter(trained_model.fc_dis1.weight.t().clone())
        self.b_dis1 = nn.Parameter(trained_model.fc_dis1.bias.clone())
        self.w_dis2 = nn.Parameter(trained_model.fc_dis2.weight.t().clone())
        self.b_dis2 = nn.Parameter(trained_model.fc_dis2.bias.clone())
        
        self.w_comp1 = nn.Parameter(trained_model.fc_comp1.weight.t().clone())
        self.b_comp1 = nn.Parameter(trained_model.fc_comp1.bias.clone())
        self.w_comp2 = nn.Parameter(trained_model.fc_comp2.weight.t().clone())
        self.b_comp2 = nn.Parameter(trained_model.fc_comp2.bias.clone())
        
        self.w_urg1 = nn.Parameter(trained_model.fc_urg1.weight.t().clone())
        self.b_urg1 = nn.Parameter(trained_model.fc_urg1.bias.clone())
        self.w_urg2 = nn.Parameter(trained_model.fc_urg2.weight.t().clone())
        self.b_urg2 = nn.Parameter(trained_model.fc_urg2.bias.clone())

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        
        d = torch.matmul(cls_rep, self.w_dis1) + self.b_dis1
        logits_disease = torch.matmul(torch.relu(d), self.w_dis2) + self.b_dis2
        
        c = torch.matmul(cls_rep, self.w_comp1) + self.b_comp1
        logits_complaint = torch.matmul(torch.relu(c), self.w_comp2) + self.b_comp2
        
        u = torch.matmul(cls_rep, self.w_urg1) + self.b_urg1
        logits_urgency = torch.matmul(torch.relu(u), self.w_urg2) + self.b_urg2
        
        return logits_disease, logits_complaint, logits_urgency

# ──────────────────────────────────────────────
# 3. EVALUATION & BOOTSTRAP UTILITIES
# ──────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=10):
    probs, labels = np.array(probs), np.array(labels)
    preds, confs = np.argmax(probs, axis=1), np.max(probs, axis=1)
    ece, bin_boundaries = 0.0, np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i+1])
        if np.mean(in_bin) > 0:
            ece += np.mean(in_bin) * np.abs(np.mean(preds[in_bin] == labels[in_bin]) - np.mean(confs[in_bin]))
    return float(ece)

def optimize_disease_thresholds(y_true, y_probs):
    thresholds = []
    for c in range(len(PULM_DISEASE_CLASSES)):
        best_f1 = -1.0
        best_th = 0.50
        for th in np.linspace(0.10, 0.90, 41):
            preds = (y_probs[:, c] >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true[:, c], preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        thresholds.append(best_th)
    return np.array(thresholds)

def bootstrap_ci(trues, preds, metric_fn, n_bootstraps=500, alpha=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(trues), len(trues))
        score = metric_fn(trues[idx], preds[idx])
        bootstrapped_scores.append(score)
    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int((1.0 - alpha) / 2.0 * n_bootstraps)]
    upper = sorted_scores[int((1.0 + alpha) / 2.0 * n_bootstraps)]
    return float(lower), float(upper)

def evaluate_pulmonologist(model, loader, device, disease_thresholds=None, desc="Evaluation"):
    log_step(f"Executing {desc} on {len(loader.dataset)} samples...")
    model.eval()
    
    d_trues, d_probs = [], []
    c_trues, c_preds, c_probs = [], [], []
    u_trues, u_preds, u_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            d_t = batch['disease_labels'].numpy()
            c_t = batch['complaint_label'].numpy()
            u_t = batch['urgency_label'].numpy()
            
            d_l, c_l, u_l = model(ids, mask)
            
            d_trues.append(d_t)
            d_probs.append(torch.sigmoid(d_l).cpu().numpy())
            
            c_trues.extend(c_t.tolist())
            c_preds.extend(torch.argmax(c_l, dim=1).cpu().numpy().tolist())
            c_probs.append(torch.softmax(c_l, dim=1).cpu().numpy())
            
            u_trues.extend(u_t.tolist())
            u_preds.extend(torch.argmax(u_l, dim=1).cpu().numpy().tolist())
            u_probs.append(torch.softmax(u_l, dim=1).cpu().numpy())
            
    d_trues = np.vstack(d_trues)
    d_probs = np.vstack(d_probs)
    c_trues = np.array(c_trues)
    c_preds = np.array(c_preds)
    c_probs = np.vstack(c_probs)
    u_trues = np.array(u_trues)
    u_preds = np.array(u_preds)
    u_probs = np.vstack(u_probs)
    
    if disease_thresholds is None:
        disease_thresholds = np.full(len(PULM_DISEASE_CLASSES), 0.50)
        
    d_preds = (d_probs >= disease_thresholds).astype(int)
    
    # Disease Metrics
    d_per_class = {}
    f1_list, prec_list, rec_list = [], [], []
    for i, d_name in enumerate(PULM_DISEASE_CLASSES):
        p, r, f1, s = precision_recall_fscore_support(d_trues[:, i], d_preds[:, i], average='binary', zero_division=0)
        try: auc = roc_auc_score(d_trues[:, i], d_probs[:, i])
        except Exception: auc = 0.5
        try: ap = average_precision_score(d_trues[:, i], d_probs[:, i])
        except Exception: ap = 0.0
            
        d_per_class[d_name] = {
            "precision": float(p), "recall": float(r), "f1_score": float(f1),
            "support": int(d_trues[:, i].sum()), "auroc": float(auc), "auprc": float(ap),
            "threshold": float(disease_thresholds[i])
        }
        f1_list.append(f1)
        prec_list.append(p)
        rec_list.append(r)
        
    d_macro_f1 = float(np.mean(f1_list))
    d_micro_p, d_micro_r, d_micro_f1, _ = precision_recall_fscore_support(d_trues, d_preds, average='micro', zero_division=0)
    d_weighted_p, d_weighted_r, d_weighted_f1, _ = precision_recall_fscore_support(d_trues, d_preds, average='weighted', zero_division=0)
    d_hamming = float(hamming_loss(d_trues, d_preds))
    d_subset_acc = float(accuracy_score(d_trues, d_preds))
    d_macro_auroc = float(np.mean([v["auroc"] for v in d_per_class.values()]))
    d_macro_auprc = float(np.mean([v["auprc"] for v in d_per_class.values()]))
    
    # Complaint Metrics
    c_acc = float(np.mean(c_trues == c_preds))
    cp_m, cr_m, cf_m, _ = precision_recall_fscore_support(c_trues, c_preds, labels=range(11), average='macro', zero_division=0)
    cp_w, cr_w, cf_w, _ = precision_recall_fscore_support(c_trues, c_preds, labels=range(11), average='weighted', zero_division=0)
    c_ece = compute_ece(c_probs, c_trues)
    c_cm = confusion_matrix(c_trues, c_preds, labels=range(11)).tolist()
    
    # Urgency Metrics
    u_acc = float(np.mean(u_trues == u_preds))
    up_m, ur_m, uf_m, _ = precision_recall_fscore_support(u_trues, u_preds, labels=range(3), average='macro', zero_division=0)
    up_w, ur_w, uf_w, _ = precision_recall_fscore_support(u_trues, u_preds, labels=range(3), average='weighted', zero_division=0)
    u_ece = compute_ece(u_probs, u_trues)
    u_cm = confusion_matrix(u_trues, u_preds, labels=range(3)).tolist()
    
    return {
        "disease": {
            "macro_f1": d_macro_f1, "micro_f1": float(d_micro_f1), "weighted_f1": float(d_weighted_f1),
            "macro_precision": float(np.mean(prec_list)), "macro_recall": float(np.mean(rec_list)),
            "subset_accuracy": d_subset_acc, "hamming_loss": d_hamming,
            "macro_auroc": d_macro_auroc, "macro_auprc": d_macro_auprc,
            "per_class": d_per_class
        },
        "complaint": {
            "accuracy": c_acc, "macro_f1": float(cf_m), "weighted_f1": float(cf_w),
            "macro_precision": float(cp_m), "macro_recall": float(cr_m), "ece": float(c_ece),
            "cm": c_cm
        },
        "urgency": {
            "accuracy": u_acc, "macro_f1": float(uf_m), "weighted_f1": float(uf_w),
            "macro_precision": float(up_m), "macro_recall": float(ur_m), "ece": float(u_ece),
            "cm": u_cm
        },
        "raw_outputs": {
            "d_trues": d_trues, "d_probs": d_probs, "d_preds": d_preds,
            "c_trues": c_trues, "c_preds": c_preds, "c_probs": c_probs,
            "u_trues": u_trues, "u_preds": u_preds, "u_probs": u_probs
        }
    }

# ──────────────────────────────────────────────
# 4. MAIN TRAINING SCRIPT
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pulmonologist Specialist Training")
    parser.add_argument("--data_dir", type=str, default="datasets/processed/pulmonologist")
    parser.add_argument("--output_dir", type=str, default="experiments/pulmonologist_mimic_final")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("backend/models", exist_ok=True)
    
    print("=== STARTING PULMONOLOGIST SPECIALIST GPU TRAINING ===")
    print(f"Backbone: {args.model_name}")
    print(f"Sequence Length: {args.max_len} | Batch Size: {args.batch_size} | LR: {args.lr}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_step(f"Training Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Load Splits
    with open(os.path.join(args.data_dir, "train_split.json"), "r", encoding="utf-8") as f: train_recs = json.load(f)
    with open(os.path.join(args.data_dir, "val_split.json"), "r", encoding="utf-8") as f: val_recs = json.load(f)
    with open(os.path.join(args.data_dir, "test_split.json"), "r", encoding="utf-8") as f: test_recs = json.load(f)
    
    print(f"Cohort Scale: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)} (Total={len(train_recs)+len(val_recs)+len(test_recs)})")
    
    # Tokenize Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = PulmonologistDataset(train_recs, tokenizer, max_len=args.max_len, desc="Train Split")
    val_dataset = PulmonologistDataset(val_recs, tokenizer, max_len=args.max_len, desc="Val Split")
    test_dataset = PulmonologistDataset(test_recs, tokenizer, max_len=args.max_len, desc="Test Split")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*2, pin_memory=(device.type == 'cuda'))
    
    # Class Weights from TRAIN ONLY
    d_matrix = np.array(train_dataset.disease_targets)
    pos_counts = d_matrix.sum(axis=0)
    neg_counts = len(d_matrix) - pos_counts
    pos_weights = np.sqrt(neg_counts / np.maximum(pos_counts, 1))
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float).to(device)
    
    crit_disease = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    
    c_counts = np.bincount(train_dataset.complaint_targets, minlength=11)
    c_weights = len(train_dataset) / (11.0 * np.maximum(c_counts, 1).astype(float))
    crit_complaint = nn.CrossEntropyLoss(weight=torch.tensor(c_weights, dtype=torch.float).to(device))
    
    u_counts = np.bincount(train_dataset.urgency_targets, minlength=3)
    u_weights = len(train_dataset) / (3.0 * np.maximum(u_counts, 1).astype(float))
    crit_urgency = nn.CrossEntropyLoss(weight=torch.tensor(u_weights, dtype=torch.float).to(device))
    
    # Model & Optimizer
    model = PulmonologistSpecialistModel(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_comb_f1 = -1.0
    best_thresholds = None
    validation_history = []
    
    start_train = time.time()
    log_step(f"Starting Multi-Task Training Loop ({args.epochs} epochs)...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            d_t = batch['disease_labels'].to(device)
            c_t = batch['complaint_label'].to(device)
            u_t = batch['urgency_label'].to(device)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    d_l, c_l, u_l = model(ids, mask)
                    loss_d = crit_disease(d_l, d_t)
                    loss_c = crit_complaint(c_l, c_t)
                    loss_u = crit_urgency(u_l, u_t)
                    loss = 1.0 * loss_d + 0.7 * loss_c + 0.5 * loss_u
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                d_l, c_l, u_l = model(ids, mask)
                loss_d = crit_disease(d_l, d_t)
                loss_c = crit_complaint(c_l, c_t)
                loss_u = crit_urgency(u_l, u_t)
                loss = 1.0 * loss_d + 0.7 * loss_c + 0.5 * loss_u
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            scheduler.step()
            total_loss += loss.item()
            
            if (step + 1) % 500 == 0 or step == len(train_loader) - 1:
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")
                
        # Validation Evaluation & Threshold Search (VALIDATION SET ONLY)
        val_eval_raw = evaluate_pulmonologist(model, val_loader, device, disease_thresholds=None, desc=f"Validation (Epoch {epoch+1})")
        opt_th = optimize_disease_thresholds(val_eval_raw["raw_outputs"]["d_trues"], val_eval_raw["raw_outputs"]["d_probs"])
        val_eval_opt = evaluate_pulmonologist(model, val_loader, device, disease_thresholds=opt_th, desc=f"Validation Optimized (Epoch {epoch+1})")
        
        val_d_f1 = val_eval_opt["disease"]["macro_f1"]
        val_c_f1 = val_eval_opt["complaint"]["macro_f1"]
        val_u_f1 = val_eval_opt["urgency"]["macro_f1"]
        val_comb_f1 = (val_d_f1 + val_c_f1 + val_u_f1) / 3.0
        
        val_record = {
            "epoch": epoch + 1,
            "disease_macro_f1": val_d_f1,
            "disease_micro_f1": val_eval_opt["disease"]["micro_f1"],
            "complaint_macro_f1": val_c_f1,
            "urgency_macro_f1": val_u_f1,
            "combined_macro_f1": val_comb_f1,
            "optimal_thresholds": opt_th.tolist()
        }
        validation_history.append(val_record)
        log_step(f"Epoch {epoch+1} Val Summary: Dis Macro-F1={val_d_f1:.4f} (Micro-F1={val_eval_opt['disease']['micro_f1']:.4f}) | Comp Macro-F1={val_c_f1:.4f} | Urg Macro-F1={val_u_f1:.4f} | Comb F1={val_comb_f1:.4f}")
        
        if val_comb_f1 > best_val_comb_f1:
            best_val_comb_f1 = val_comb_f1
            best_thresholds = opt_th
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            log_step(f"Saved new best checkpoint with Combined F1={val_comb_f1:.4f} to {ckpt_path}")
            
    training_time_min = (time.time() - start_train) / 60.0
    
    # Save Thresholds & Class Weights
    thresholds_dict = {PULM_DISEASE_CLASSES[i]: float(best_thresholds[i]) for i in range(len(PULM_DISEASE_CLASSES))}
    with open(os.path.join(args.output_dir, "thresholds.json"), "w", encoding="utf-8") as f: json.dump(thresholds_dict, f, indent=2)
    with open(os.path.join(args.output_dir, "class_weights.json"), "w", encoding="utf-8") as f: json.dump(pos_weights.tolist(), f, indent=2)
    with open(os.path.join(args.output_dir, "validation_results.json"), "w", encoding="utf-8") as f: json.dump(validation_history, f, indent=2)
    
    # ──────────────────────────────────────────────
    # 5. EXACTLY ONE TEST EVALUATION
    # ──────────────────────────────────────────────
    log_step("Executing EXACTLY ONE Final Evaluation on Untouched TEST Split (14,512 records)...")
    ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    test_results = evaluate_pulmonologist(model, test_loader, device, disease_thresholds=best_thresholds, desc="Final Test Set Evaluation")
    
    # 95% Bootstrap Confidence Intervals
    d_trues_arr = test_results["raw_outputs"]["d_trues"]
    d_preds_arr = test_results["raw_outputs"]["d_preds"]
    d_probs_arr = test_results["raw_outputs"]["d_probs"]
    
    def calc_micro_f1(y_t, y_p):
        _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average='micro', zero_division=0)
        return f1
    def calc_macro_f1(y_t, y_p):
        _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
        return f1
    def calc_macro_auroc(y_t, y_probs):
        aurocs = []
        for i in range(len(PULM_DISEASE_CLASSES)):
            if len(np.unique(y_t[:, i])) > 1:
                aurocs.append(roc_auc_score(y_t[:, i], y_probs[:, i]))
        return np.mean(aurocs) if aurocs else 0.5
        
    micro_ci = bootstrap_ci(d_trues_arr, d_preds_arr, calc_micro_f1)
    macro_ci = bootstrap_ci(d_trues_arr, d_preds_arr, calc_macro_f1)
    auroc_ci = bootstrap_ci(d_trues_arr, d_probs_arr, calc_macro_auroc)
    
    ci_dict = {
        "disease_micro_f1_95ci": micro_ci,
        "disease_macro_f1_95ci": macro_ci,
        "disease_macro_auroc_95ci": auroc_ci
    }
    with open(os.path.join(args.output_dir, "confidence_intervals.json"), "w", encoding="utf-8") as f:
        json.dump(ci_dict, f, indent=2)
        
    # Save Per-Class CSV
    per_class_rows = []
    for d_name, metrics in test_results["disease"]["per_class"].items():
        per_class_rows.append({
            "Class": d_name,
            "Type": "Pulmonary_Disease",
            "Support": metrics["support"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1_score"],
            "AUROC": metrics["auroc"],
            "AUPRC": metrics["auprc"],
            "Frozen_Threshold": metrics["threshold"]
        })
    df_per_class = pd.DataFrame(per_class_rows)
    df_per_class.to_csv(os.path.join(args.output_dir, "per_class_metrics.csv"), index=False)
    
    # ──────────────────────────────────────────────
    # 6. ONNX EXPORT & LATENCY BENCHMARK
    # ──────────────────────────────────────────────
    log_step("Exporting Pulmonologist Model to ONNX Graph...")
    onnx_model = PulmonologistSpecialistONNX(model).to(device)
    onnx_model.eval()
    
    dummy_ids = torch.ones(1, args.max_len, dtype=torch.long).to(device)
    dummy_mask = torch.ones(1, args.max_len, dtype=torch.long).to(device)
    
    onnx_fp32_path = "backend/models/pulmonologist_final.onnx"
    onnx_int8_path = "backend/models/pulmonologist_final_quant.onnx"
    
    torch.onnx.export(
        onnx_model,
        (dummy_ids, dummy_mask),
        onnx_fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["disease_logits", "complaint_logits", "urgency_logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}},
        opset_version=18
    )
    print(f"Exported FP32 ONNX to {onnx_fp32_path}")
    
    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QUInt8)
    print(f"Exported INT8 ONNX to {onnx_int8_path}")
    
    # CPU Latency Benchmark (100 samples)
    sess_int8 = ort.InferenceSession(onnx_int8_path, providers=['CPUExecutionProvider'])
    latencies = []
    for b in test_dataset.encodings[:100]:
        t0 = time.perf_counter()
        sess_int8.run(None, {'input_ids': b['input_ids'].numpy().reshape(1, -1), 'attention_mask': b['attention_mask'].numpy().reshape(1, -1)})
        latencies.append((time.perf_counter() - t0) * 1000)
    cpu_latency_ms = float(np.mean(latencies))
    
    # Numerical Parity Check
    sess_fp32 = ort.InferenceSession(onnx_fp32_path, providers=['CPUExecutionProvider'])
    py_d, fp32_d, int8_d = [], [], []
    model_cpu = model.to('cpu')
    model_cpu.eval()
    
    for b in test_dataset.encodings[:50]:
        inp = b['input_ids'].numpy().reshape(1, -1)
        msk = b['attention_mask'].numpy().reshape(1, -1)
        with torch.no_grad():
            py_logits, _, _ = model_cpu(b['input_ids'].unsqueeze(0), b['attention_mask'].unsqueeze(0))
            py_d.append(py_logits.numpy().flatten())
        o_d, _, _ = sess_fp32.run(None, {'input_ids': inp, 'attention_mask': msk})
        fp32_d.append(o_d.flatten())
        q_d, _, _ = sess_int8.run(None, {'input_ids': inp, 'attention_mask': msk})
        int8_d.append(q_d.flatten())
        
    py_arr, fp32_arr, int8_arr = np.array(py_d), np.array(fp32_d), np.array(int8_d)
    mae_fp32 = float(np.mean(np.abs(py_arr - fp32_arr)))
    mae_int8 = float(np.mean(np.abs(py_arr - int8_arr)))
    
    th_v = np.array([thresholds_dict[d] for d in PULM_DISEASE_CLASSES])
    agree_fp32 = float(np.mean((1.0 / (1.0 + np.exp(-py_arr)) >= th_v) == (1.0 / (1.0 + np.exp(-fp32_arr)) >= th_v)) * 100)
    agree_int8 = float(np.mean((1.0 / (1.0 + np.exp(-py_arr)) >= th_v) == (1.0 / (1.0 + np.exp(-int8_arr)) >= th_v)) * 100)
    
    # ──────────────────────────────────────────────
    # 7. GENERATE FIGURES (ROC, PR, Calibration, Confusion)
    # ──────────────────────────────────────────────
    log_step("Generating visual figures...")
    
    # ROC Curves
    plt.figure(figsize=(9, 7))
    for i, d_name in enumerate(PULM_DISEASE_CLASSES):
        fpr, tpr, _ = roc_curve(d_trues_arr[:, i], d_probs_arr[:, i])
        auc = roc_auc_score(d_trues_arr[:, i], d_probs_arr[:, i])
        plt.plot(fpr, tpr, label=f"{d_name} (AUROC = {auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(f"Pulmonologist Specialist: Multi-Label Disease ROC Curves\n(Macro-AUROC: {test_results['disease']['macro_auroc']:.4f})", fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "roc_curves.png"), dpi=200)
    plt.close()
    
    # PR Curves
    plt.figure(figsize=(9, 7))
    for i, d_name in enumerate(PULM_DISEASE_CLASSES):
        prec, rec, _ = precision_recall_curve(d_trues_arr[:, i], d_probs_arr[:, i])
        ap = average_precision_score(d_trues_arr[:, i], d_probs_arr[:, i])
        plt.plot(rec, prec, label=f"{d_name} (AUPRC = {ap:.3f})", lw=2)
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.title(f"Pulmonologist Specialist: Multi-Label Precision-Recall Curves\n(Macro-AUPRC: {test_results['disease']['macro_auprc']:.4f})", fontsize=13)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pr_curves.png"), dpi=200)
    plt.close()
    
    # Confusion & Metric Plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(PULM_DISEASE_CLASSES))
    width = 0.35
    f1s = [metrics["f1_score"] for metrics in test_results["disease"]["per_class"].values()]
    aucs = [metrics["auroc"] for metrics in test_results["disease"]["per_class"].values()]
    axes[0].bar(x - width/2, f1s, width, label='F1-Score', color='royalblue')
    axes[0].bar(x + width/2, aucs, width, label='AUROC', color='cornflowerblue')
    axes[0].set_title(f"Pulmonary Disease Metrics\n(Micro-F1: {test_results['disease']['micro_f1']:.3f} | Macro-F1: {test_results['disease']['macro_f1']:.3f})", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("_", "\n") for c in PULM_DISEASE_CLASSES], fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    comp_short = [c[:6] for c in PULM_COMPLAINT_CLASSES]
    sns.heatmap(np.array(test_results["complaint"]["cm"]), annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=comp_short, yticklabels=comp_short)
    axes[1].set_title(f"Chief Complaint Confusion Matrix\n(Macro-F1: {test_results['complaint']['macro_f1']:.3f} | Acc: {test_results['complaint']['accuracy']*100:.1f}%)", fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.heatmap(np.array(test_results["urgency"]["cm"]), annot=True, fmt='d', cmap='Greens', ax=axes[2], xticklabels=URGENCY_CLASSES, yticklabels=URGENCY_CLASSES)
    axes[2].set_title(f"Clinical Urgency Confusion Matrix\n(Acc: {test_results['urgency']['accuracy']*100:.1f}% | ECE: {test_results['urgency']['ece']:.4f})", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrices.png"), dpi=200)
    plt.close()
    
    # ──────────────────────────────────────────────
    # 8. SAVE MANIFESTS & FINAL REPORT
    # ──────────────────────────────────────────────
    clean_test_metrics = {k: v for k, v in test_results.items() if k != "raw_outputs"}
    clean_test_metrics["disease"]["micro_f1_95ci"] = micro_ci
    clean_test_metrics["disease"]["macro_f1_95ci"] = macro_ci
    clean_test_metrics["disease"]["macro_auroc_95ci"] = auroc_ci
    
    metrics_manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "specialist": "Pulmonologist_Specialist_Final",
        "backbone": args.model_name,
        "cohort_counts": {
            "train": len(train_recs), "val": len(val_recs), "test": len(test_recs), "total": len(train_recs)+len(val_recs)+len(test_recs)
        },
        "test_metrics": clean_test_metrics,
        "onnx": {
            "fp32_size_mb": os.path.getsize(onnx_fp32_path) / (1024**2),
            "int8_size_mb": os.path.getsize(onnx_int8_path) / (1024**2),
            "cpu_latency_ms": cpu_latency_ms,
            "fp32_parity_mae": mae_fp32,
            "fp32_agreement_pct": agree_fp32,
            "int8_parity_mae": mae_int8,
            "int8_agreement_pct": agree_int8
        }
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f: json.dump(metrics_manifest, f, indent=2)
    
    print("\n=======================================================")
    print("PULMONOLOGIST SPECIALIST TRAINING & EVALUATION COMPLETE")
    print(f"Disease Test Micro-F1: {clean_test_metrics['disease']['micro_f1']:.4f} | Macro-F1: {clean_test_metrics['disease']['macro_f1']:.4f} | Macro-AUROC: {clean_test_metrics['disease']['macro_auroc']:.4f}")
    print(f"Complaint Test Macro-F1: {clean_test_metrics['complaint']['macro_f1']:.4f} | Accuracy: {clean_test_metrics['complaint']['accuracy']:.4f}")
    print(f"Urgency Test Macro-F1: {clean_test_metrics['urgency']['macro_f1']:.4f} | Accuracy: {clean_test_metrics['urgency']['accuracy']:.4f}")
    print(f"INT8 ONNX CPU Latency: {cpu_latency_ms:.2f} ms/sample")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
