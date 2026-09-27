import os
import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, hamming_loss, accuracy_score

from src.training.train_cardiologist import CardiologistSpecialistModel, CardiologistDataset, DISEASE_CLASSES, COMPLAINT_CLASSES, URGENCY_CLASSES
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def run_authoritative_evaluation(ckpt_path="experiments/cardiologist_mimic_final/best_model.pt",
                                thresholds_path="experiments/cardiologist_mimic_final/thresholds.json",
                                test_split_path="datasets/processed/common/test_split.json",
                                device_name="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_name)
    with open(test_split_path, "r", encoding="utf-8") as f: test_recs = json.load(f)
    with open(thresholds_path, "r", encoding="utf-8") as f: thresholds_dict = json.load(f)
    th_vec = np.array([thresholds_dict[d] for d in DISEASE_CLASSES])
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    dataset = CardiologistDataset(test_recs, tokenizer, max_len=384)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = CardiologistSpecialistModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    d_trues, d_probs = [], []
    c_trues, c_preds = [], []
    u_trues, u_preds = [], []
    
    with torch.no_grad():
        for b in loader:
            ids = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            d_l, c_l, u_l = model(ids, mask)
            
            d_trues.append(b['disease_labels'].numpy())
            d_probs.append(torch.sigmoid(d_l).cpu().numpy())
            c_trues.extend(b['complaint_label'].numpy().tolist())
            c_preds.extend(torch.argmax(c_l, dim=1).cpu().numpy().tolist())
            u_trues.extend(b['urgency_label'].numpy().tolist())
            u_preds.extend(torch.argmax(u_l, dim=1).cpu().numpy().tolist())
            
    d_trues = np.vstack(d_trues)
    d_probs = np.vstack(d_probs)
    c_trues = np.array(c_trues)
    c_preds = np.array(c_preds)
    u_trues = np.array(u_trues)
    u_preds = np.array(u_preds)
    
    d_preds = (d_probs >= th_vec).astype(int)
    
    _, _, d_micro, _ = precision_recall_fscore_support(d_trues, d_preds, average='micro', zero_division=0)
    _, _, d_macro, _ = precision_recall_fscore_support(d_trues, d_preds, average='macro', zero_division=0)
    aurocs = [roc_auc_score(d_trues[:, i], d_probs[:, i]) for i in range(6)]
    d_auroc = float(np.mean(aurocs))
    
    _, _, c_macro, _ = precision_recall_fscore_support(c_trues, c_preds, labels=range(11), average='macro', zero_division=0)
    u_acc = float(accuracy_score(u_trues, u_preds))
    
    return {
        "disease_micro_f1": float(d_micro),
        "disease_macro_f1": float(d_macro),
        "disease_macro_auroc": d_auroc,
        "complaint_macro_f1": float(c_macro),
        "urgency_accuracy": u_acc
    }

if __name__ == "__main__":
    print("evaluate_cardiologist module ready.")
