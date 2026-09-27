import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support

DISEASE_CLASSES = ["angina", "arrhythmia", "atrial fibrillation", "coronary artery disease", "heart failure", "hypertension"]
COMPLAINT_CLASSES = ["cardiac assessment", "chest pain", "cough", "dyspnea", "edema", "fatigue", "headache", "hypertensive symptoms", "nausea", "palpitations", "syncope"]
URGENCY_CLASSES = ["Emergency", "Urgent", "Routine"]

DISEASE_MAP = {lbl: idx for idx, lbl in enumerate(DISEASE_CLASSES)}
COMPLAINT_MAP = {lbl: idx for idx, lbl in enumerate(COMPLAINT_CLASSES)}
URGENCY_MAP = {lbl: idx for idx, lbl in enumerate(URGENCY_CLASSES)}

class CardiologistDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=384):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = [r.get("text", "No clinical narrative.") for r in records]
        
        self.disease_targets = []
        for r in records:
            vec = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
            for d in r.get("cardio_disease_labels", []):
                if d in DISEASE_MAP: vec[DISEASE_MAP[d]] = 1.0
            if vec.sum() == 0: vec[DISEASE_MAP["hypertension"]] = 1.0
            self.disease_targets.append(vec)
            
        self.complaint_targets = [COMPLAINT_MAP.get(r.get("chief_complaint", "cardiac assessment"), 0) for r in records]
        self.urgency_targets = [URGENCY_MAP.get(r.get("urgency", "Emergency"), 0) for r in records]
        
        self.encodings = []
        for text in self.texts:
            enc = tokenizer(str(text), add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            self.encodings.append({'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()})

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        item = self.encodings[idx].copy()
        item['disease_labels'] = torch.tensor(self.disease_targets[idx], dtype=torch.float)
        item['complaint_label'] = torch.tensor(self.complaint_targets[idx], dtype=torch.long)
        item['urgency_label'] = torch.tensor(self.urgency_targets[idx], dtype=torch.long)
        return item

class CardiologistSpecialistModel(nn.Module):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        h = self.transformer.config.hidden_size
        self.fc_dis1 = nn.Linear(h, 256)
        self.fc_dis2 = nn.Linear(256, 6)
        self.fc_comp1 = nn.Linear(h, 256)
        self.fc_comp2 = nn.Linear(256, 11)
        self.fc_urg1 = nn.Linear(h, 256)
        self.fc_urg2 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        rep = out[0][:, 0, :]
        d = self.dropout(self.relu(self.fc_dis1(rep)))
        c = self.dropout(self.relu(self.fc_comp1(rep)))
        u = self.dropout(self.relu(self.fc_urg1(rep)))
        return self.fc_dis2(d), self.fc_comp2(c), self.fc_urg2(u)

def optimize_thresholds(y_true, y_probs):
    thresholds = []
    for c in range(len(DISEASE_CLASSES)):
        best_f1, best_th = -1.0, 0.50
        for th in np.linspace(0.10, 0.90, 41):
            preds = (y_probs[:, c] >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true[:, c], preds, average='binary', zero_division=0)
            if f1 > best_f1: best_f1, best_th = f1, th
        thresholds.append(best_th)
    return np.array(thresholds)

print("train_cardiologist module ready.")
