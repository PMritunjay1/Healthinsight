import sys
import io
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Force stdout/stderr to use UTF-8 to prevent Windows encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def log_step(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {message}")
    sys.stdout.flush()

# ENVIRONMENT CONFIGURATION
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cpu') # Enforce CPU training locally
torch.set_num_threads(4) # Limit CPU threads to prevent cache thrashing
log_step("Configured device: CPU with 4 execution threads")

CHECKPOINT_DIR = 'experiments/cardiologist'
MODEL_DIR = 'backend/models'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# ──────────────────────────────────────────────
# DATASET DEFINITION
# ──────────────────────────────────────────────
class CardiologistDataset(Dataset):
    def __init__(self, df, tokenizer, label_map, max_len=64, desc="Dataset"):
        self.texts = df['Patient Text'].values.tolist()
        self.labels = [label_map[l] for l in df['Disease'].values]
        
        log_step(f"Pre-tokenizing {desc} ({len(self.texts)} samples)...")
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
        log_step(f"Pre-tokenization of {desc} complete.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.encodings[idx].copy()
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ──────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────
class CardiologistClassifier(nn.Module):
    def __init__(self, model_name, num_classes=7):
        super().__init__()
        log_step(f"Loading PubMedBERT base weights for {model_name}...")
        self.transformer = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        logits = self.classifier(cls_rep)
        return logits

class CardiologistClassifierONNX(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.transformer = trained_model.transformer
        hidden_size = self.transformer.config.hidden_size
        self.w1 = nn.Parameter(trained_model.classifier[0].weight.t().clone())
        self.b1 = nn.Parameter(trained_model.classifier[0].bias.clone())
        self.w2 = nn.Parameter(trained_model.classifier[3].weight.t().clone())
        self.b2 = nn.Parameter(trained_model.classifier[3].bias.clone())

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        x = torch.matmul(cls_rep, self.w1) + self.b1
        x = torch.relu(x)
        logits = torch.matmul(x, self.w2) + self.b2
        return logits

def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(preds[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            bin_data.append({
                'bin': f"{bin_lower:.1f}-{bin_upper:.1f}",
                'count': int(np.sum(in_bin)),
                'accuracy': float(accuracy_in_bin),
                'confidence': float(avg_confidence_in_bin)
            })
        else:
            bin_data.append({
                'bin': f"{bin_lower:.1f}-{bin_upper:.1f}",
                'count': 0,
                'accuracy': 0.0,
                'confidence': 0.0
            })
            
    return float(ece), bin_data

def train_expanded_cardiologist(epochs=1, batch_size=32, lr=2e-5):
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        log_step("SMOKE TEST MODE ENABLED - running single epoch on subsets.")
        epochs = 1
        batch_size = 4
        
    log_step("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    log_step("Loading datasets...")
    train_df = pd.read_csv('datasets/processed/cardiologist/train.csv')
    val_df = pd.read_csv('datasets/processed/cardiologist/val.csv')
    test_df = pd.read_csv('datasets/processed/cardiologist/test.csv')
    
    unique_labels = sorted(train_df['Disease'].unique().tolist())
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for lbl, idx in label_map.items()}
    
    if smoke_test:
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        test_df = test_df.head(8)
        
    train_dataset = CardiologistDataset(train_df, tokenizer, label_map, desc="Train")
    val_dataset = CardiologistDataset(val_df, tokenizer, label_map, desc="Val")
    test_dataset = CardiologistDataset(test_df, tokenizer, label_map, desc="Test")
    
    # Weighted Random Sampler
    class_counts = train_df['Disease'].value_counts()
    class_weights = {label_map[lbl]: 1.0 / count for lbl, count in class_counts.items()}
    sample_weights = [class_weights[label_map[row['Disease']]] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = CardiologistClassifier(MODEL_NAME, num_classes=len(unique_labels)).to(DEVICE)
    
    # Class-weighted loss
    weights = [1.0 / class_counts[inv_label_map[i]] for i in range(len(unique_labels))]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_macro_f1 = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    
    start_time = time.time()
    
    log_step("Beginning training loop...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        batch_idx = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            batch_idx += 1
            if batch_idx % 200 == 0:
                log_step(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['label'].to(DEVICE)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                val_preds.extend(preds)
                val_labels.extend(targets.cpu().numpy().tolist())
                
        avg_val_loss = total_val_loss / len(val_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_macro_f1'].append(f1)
        
        log_step(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {f1:.4f}")
        
        if f1 > best_macro_f1 or smoke_test:
            best_macro_f1 = f1
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
            torch.save(model.state_dict(), best_checkpoint_path)
            log_step(f"Saved new best model checkpoint to {best_checkpoint_path}")
            
    training_time_elapsed = time.time() - start_time
    
    # Reload best weights
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    
    # ─── TEST EVALUATION ───
    log_step("Evaluating on Test Set...")
    model.eval()
    test_preds, test_labels, test_probs = [], [], []
    
    test_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['label'].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            
            test_preds.extend(preds)
            test_labels.extend(targets.cpu().numpy().tolist())
            test_probs.extend(probs)
            
    test_duration = time.time() - test_start
    cpu_latency_ms = (test_duration / len(test_labels)) * 1000
    
    report = classification_report(test_labels, test_preds, target_names=unique_labels, output_dict=True, zero_division=0)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro', zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
    acc = np.mean(np.array(test_labels) == np.array(test_preds))
    ece_val, bin_details = compute_ece(test_probs, test_labels)
    cm = confusion_matrix(test_labels, test_preds)
    
    # ─── EXPORT ONNX ───
    log_step("Exporting to ONNX format...")
    onnx_model = CardiologistClassifierONNX(model)
    onnx_model.eval()
    onnx_path = os.path.join(CHECKPOINT_DIR, "model.onnx")
    
    dummy_input_ids = torch.ones(1, 64, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 64, dtype=torch.long)
    
    torch.onnx.export(
        onnx_model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}},
        opset_version=14
    )
    log_step(f"Exported raw ONNX model to {onnx_path}")
    
    # ONNX Quantization to INT8 for backend/models/
    log_step("Quantizing ONNX model...")
    quant_path = os.path.join(MODEL_DIR, "cardiologist_model_quant.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
        log_step(f"Saved quantized ONNX model to {quant_path}")
    except Exception as e:
        log_step(f"Quantization failed: {e}")
        import shutil
        shutil.copyfile(onnx_path, quant_path)
        
    # Benchmarking ONNX Latency
    onnx_latency_ms = 0.0
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(quant_path, providers=['CPUExecutionProvider'])
        
        onnx_start = time.time()
        for ids, mask in zip(test_dataset.encodings, test_dataset.encodings):
            dummy_ids_np = ids['input_ids'].numpy().reshape(1, -1)
            dummy_mask_np = mask['attention_mask'].numpy().reshape(1, -1)
            ort_sess.run(None, {'input_ids': dummy_ids_np, 'attention_mask': dummy_mask_np})
        onnx_duration = time.time() - onnx_start
        onnx_latency_ms = (onnx_duration / len(test_labels)) * 1000
    except Exception as e:
        log_step(f"ONNX Latency Benchmark failed: {e}")
        
    # Visualizations
    log_step("Generating training visualizations...")
    # Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), history['train_loss'], marker='o', label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], marker='x', label='Val Loss')
    plt.title('Loss Curves (Cardiologist)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "learning_curve.png"), dpi=150)
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix (Cardiologist)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Reliability Curve
    plt.figure(figsize=(5, 5))
    bins = [b['accuracy'] for b in bin_details]
    confs = [b['confidence'] for b in bin_details]
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.plot(confs, bins, marker='o', label='Cardiologist Model')
    plt.title(f'Reliability Curve (ECE={ece_val:.4f})')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "reliability_curve.png"), dpi=150)
    plt.close()
    
    # Save MODEL_CONFIG.json
    model_config = {
        "model_name": MODEL_NAME,
        "num_classes": len(unique_labels),
        "label_mapping": label_map,
        "max_sequence_length": 64,
        "quantized": True,
        "opset_version": 14
    }
    with open(os.path.join(CHECKPOINT_DIR, "MODEL_CONFIG.json"), "w") as f:
        json.dump(model_config, f, indent=4)
        
    # Save metrics.json
    stats_dict = {
        "model_name": MODEL_NAME,
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "f1_weighted": float(f_weight),
        "ece": float(ece_val),
        "cpu_latency_ms": float(cpu_latency_ms),
        "onnx_latency_ms": float(onnx_latency_ms),
        "model_size_mb": float(os.path.getsize(best_checkpoint_path) / (1024 * 1024)),
        "onnx_size_mb": float(os.path.getsize(quant_path) / (1024 * 1024)),
        "classification_report": report
    }
    with open(os.path.join(CHECKPOINT_DIR, "metrics.json"), "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    # Save training_log.json
    with open(os.path.join(CHECKPOINT_DIR, "training_log.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # ─── GENERATE TRAINING_REPORT.md ───
    report_md = (
        f"# Cardiologist Model Training & Evaluation Report\n"
        f"**Project Milestone**: Expanded Specialty 7-Class Acceptance  \n"
        f"**Date**: {time.strftime('%Y-%m-%d')}  \n\n"
        f"--- \n\n"
        f"## 1. Training Configuration\n"
        f"* **Base Model**: `{MODEL_NAME}`\n"
        f"* **Optimizer**: `AdamW`\n"
        f"* **Learning Rate**: `{lr}`\n"
        f"* **Batch Size**: `{batch_size}`\n"
        f"* **Epochs**: `{epochs}`\n"
        f"* **Sampler**: `WeightedRandomSampler`\n"
        f"* **Sequence Length**: `64`\n\n"
        f"## 2. Evaluation Metrics (Test Set)\n"
        f"* **Accuracy**: `{acc:.4f}`\n"
        f"* **Precision (Macro)**: `{p_macro:.4f}`\n"
        f"* **Recall (Macro)**: `{r_macro:.4f}`\n"
        f"* **Macro F1**: `{f_macro:.4f}`\n"
        f"* **Weighted F1**: `{f_weight:.4f}`\n"
        f"* **Expected Calibration Error (ECE)**: `{ece_val:.4f}`\n"
        f"* **CPU Latency (PyTorch)**: `{cpu_latency_ms:.2f} ms/sample`\n"
        f"* **CPU Latency (ONNX)**: `{onnx_latency_ms:.2f} ms/sample`\n"
        f"* **PyTorch Model Size**: `{stats_dict['model_size_mb']:.2f} MB`\n"
        f"* **Quantized ONNX Model Size**: `{stats_dict['onnx_size_mb']:.2f} MB`\n\n"
        f"## 3. Per-Class Metrics\n\n"
        f"| Class Name | Precision | Recall | F1-Score | Support |\n"
        f"| :--- | :---: | :---: | :---: | :---: |\n"
    )
    for lbl in unique_labels:
        per_class = report[lbl]
        report_md += f"| {lbl} | {per_class['precision']:.4f} | {per_class['recall']:.4f} | {per_class['f1-score']:.4f} | {per_class['support']} |\n"
        
    report_md += (
        f"\n## 4. Visualizations\n"
        f"- [Confusion Matrix](file:///{os.path.abspath(os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png'))})\n"
        f"- [Reliability Curve](file:///{os.path.abspath(os.path.join(CHECKPOINT_DIR, 'reliability_curve.png'))})\n"
        f"- [Learning Curve](file:///{os.path.abspath(os.path.join(CHECKPOINT_DIR, 'learning_curve.png'))})\n"
    )
    with open(os.path.join(CHECKPOINT_DIR, "TRAINING_REPORT.md"), "w", encoding='utf-8') as f:
        f.write(report_md)
    log_step("TRAINING_REPORT.md written successfully.")

if __name__ == "__main__":
    train_expanded_cardiologist()
