import os
import sys
import time
import json
import argparse
import datetime
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────
# TIMESTAMP PROGRESS LOGGER
# ──────────────────────────────────────────────
def log_step(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

# ──────────────────────────────────────────────
# DATASET IMPLEMENTATION
# ──────────────────────────────────────────────
class RiskDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128, smoke_test=False, desc="dataset"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        df = pd.read_csv(csv_path)
        if smoke_test:
            # Process a small subset for CPU validation
            n_samples = 16 if "train" in csv_path else 8
            df = df.head(n_samples)
            log_step(f"!!! Smoke Test Mode !!! Truncating {desc} to {n_samples} samples.")
            
        self.texts = df['cleaned_text'].astype(str).tolist()
        self.labels = df['triage_level'].astype(int).tolist()
        
        log_step(f"Pre-tokenizing {desc} ({len(self.texts)} samples)...")
        self.encodings = []
        for idx, text in enumerate(self.texts):
            enc = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
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
class RiskClassifierModel(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super().__init__()
        log_step(f"!!! HF WEIGHT DOWNLOAD START: Loading base weights for {model_name} from Hugging Face Hub (approx. 440MB if not cached) !!!")
        self.transformer = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        log_step(f"Clinical base weights successfully loaded for {model_name}.")
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, output_attentions=False):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        # Class token CLS extraction
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        logits = self.classifier(cls_rep)
        
        if output_attentions:
            return logits, outputs.attentions
        return logits

class RiskModelONNX(nn.Module):
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

# ──────────────────────────────────────────────
# EXPECTED CALIBRATION ERROR (ECE)
# ──────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(preds[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return float(ece)

# ──────────────────────────────────────────────
# EVALUATION IMPLEMENTATION
# ──────────────────────────────────────────────
def evaluate_model(model, tokenizer, model_name, val_loader, checkpoint_dir, smoke_test=False):
    model.eval()
    device = next(model.parameters()).device
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate Softmax probs
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    
    # Calculate Metrics
    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average='macro')
    weighted_f1 = f1_score(all_labels, preds, average='weighted')
    prec = precision_score(all_labels, preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, preds, average='weighted', zero_division=0)
    ece_val = compute_ece(probs, all_labels)
    
    model_short = model_name.split('/')[-1]
    
    log_step(f"[{model_short}] Evaluation complete. Accuracy: {acc:.4f}, ECE: {ece_val:.4f}")
    
    # Generate curves & reports only outside smoke tests
    if not smoke_test:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Confusion matrix
        plt.figure(figsize=(6, 5))
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(all_labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low', 'Medium', 'High', 'Critical'],
                    yticklabels=['Low', 'Medium', 'High', 'Critical'])
        plt.title(f'Confusion Matrix: {model_short}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/confusion_matrix_{model_short}.png", dpi=200)
        plt.close()
        
        # Save validation logits/predictions JSON
        eval_log = {
            'model_name': model_name,
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'ece': float(ece_val),
            'predictions': []
        }
        
        for idx, (logit, prob, pred, label) in enumerate(zip(all_logits, probs, preds, all_labels)):
            eval_log['predictions'].append({
                'sample_index': idx,
                'raw_logits': logit.tolist(),
                'softmax_probabilities': prob.tolist(),
                'predicted_label': int(pred),
                'true_label': int(label)
            })
            
        with open(f"{checkpoint_dir}/eval_predictions_{model_short}.json", 'w') as f:
            json.dump(eval_log, f, indent=4)
            
        # Explainability sample text (attentions)
        sample_text = "patient is a 45-year-old male presenting with acute subternous crushing chest pain radiating to left arm with dyspnea."
        enc = tokenizer(sample_text, return_tensors='pt').to(device)
        model.eval()
        logits, attentions = model(enc['input_ids'], enc['attention_mask'], output_attentions=True)
        # attentions is a tuple of length layers, each layer has shape [batch, heads, seq, seq]
        last_attn = attentions[-1][0].detach().cpu().numpy() # [heads, seq, seq]
        mean_attn = np.mean(last_attn, axis=0) # [seq, seq]
        cls_attn = mean_attn[0] # Attention from CLS token to all tokens
        
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
        explain_lines = [
            f"# Self-Attention Triage Explainability - {model_short}\n",
            f"**Symptom Query**: *\"{sample_text}\"*\n",
            f"| Token | Attention Weight | Influence Rank |",
            f"| :--- | :---: | :---: |"
        ]
        
        ranked_indices = np.argsort(cls_attn)[::-1]
        for rank, rid in enumerate(ranked_indices):
            if rid < len(tokens):
                explain_lines.append(f"| `{tokens[rid]}` | {cls_attn[rid]:.6f} | {rank + 1} |")
                
        with open(f"{checkpoint_dir}/explainability_report_{model_short}.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(explain_lines))
            
        # Clinical Demo Cases Evaluation
        demo_path = 'backend/config/clinical_demo_cases.json'
        if os.path.exists(demo_path):
            with open(demo_path, 'r') as f:
                demo_cases = json.load(f)
            
            demo_preds = []
            for case in demo_cases:
                enc = tokenizer(case['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
                logits = model(enc['input_ids'], enc['attention_mask'])
                probs_case = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                pred_class = int(np.argmax(probs_case))
                
                demo_preds.append({
                    'id': case['id'],
                    'text': case['text'],
                    'predicted_triage_level': pred_class,
                    'triage_probabilities': probs_case.tolist()
                })
                
            with open(f"{checkpoint_dir}/clinical_demo_predictions_{model_short}.json", 'w') as f:
                json.dump(demo_preds, f, indent=4)

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'precision': float(prec),
        'recall': float(rec),
        'ece': float(ece_val)
    }

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train_model(model_name, smoke_test=False, checkpoint_dir='experiments/risk'):
    model_short = model_name.split('/')[-1]
    log_step(f"\n==============================================")
    log_step(f"STARTING DEVELOPMENT: {model_short}")
    log_step(f"==============================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_step(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = RiskDataset('datasets/processed/risk/train.csv', tokenizer, smoke_test=smoke_test, desc=f"Train_{model_short}")
    val_dataset = RiskDataset('datasets/processed/risk/val.csv', tokenizer, smoke_test=smoke_test, desc=f"Val_{model_short}")
    
    batch_size = 4 if smoke_test else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = RiskClassifierModel(model_name, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 1 if smoke_test else 3
    best_val_acc = 0.0
    best_checkpoint_path = f"{checkpoint_dir}/{model_short}_best.pt"
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    log_step("Beginning training loop...")
    t_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        log_step(f"Epoch {epoch+1}/{epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), best_checkpoint_path)
            log_step(f"Saved new best model checkpoint to {best_checkpoint_path}")
            
    duration = time.time() - t_start
    log_step(f"Training session complete. Duration: {duration:.2f} seconds.")
    
    # Plot curves (non smoke-test)
    if not smoke_test:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss')
        plt.plot(range(1, epochs+1), history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve - {model_short}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/loss_curve_{model_short}.png", dpi=200)
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, epochs+1), history['val_acc'], label='Val Acc', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Validation Accuracy Curve - {model_short}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/accuracy_curve_{model_short}.png", dpi=200)
        plt.close()
        
    # Load best checkpoint for evaluation
    log_step("Reloading best weights for downstream evaluations...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    log_step("Weights successfully loaded.")
    
    metrics = evaluate_model(model, tokenizer, model_name, val_loader, checkpoint_dir, smoke_test=smoke_test)
    metrics['duration_s'] = duration
    
    # ──────────────────────────────────────────────
    # ONNX EXPORT & DYNAMIC QUANTIZATION
    # ──────────────────────────────────────────────
    onnx_path = f"backend/models/risk_model_{model_short}.onnx"
    quant_path = f"backend/models/risk_model_{model_short}_quant.onnx"
    
    if not smoke_test:
        os.makedirs('backend/models', exist_ok=True)
        log_step(f"Exporting model to ONNX...")
        dummy_ids = torch.randint(0, 1000, (1, 128)).to(device)
        dummy_mask = torch.ones((1, 128)).to(device)
        
        # Use ONNX projection wrapper to bypass dynamic shape checks
        onnx_model = RiskModelONNX(model).to(device)
        onnx_model.eval()
        
        torch.onnx.export(
            onnx_model,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            opset_version=17
        )
        log_step(f"ONNX model saved to {onnx_path}")
        
        # Dynamic INT8 Quantization
        log_step(f"Applying dynamic INT8 quantization...")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8
        )
        log_step(f"Quantized ONNX model saved to {quant_path}")
        
    return metrics

# ──────────────────────────────────────────────
# ZIP PACKAGE GENERATION
# ──────────────────────────────────────────────
def create_zip_package(m1_short, m2_short):
    zip_path = 'experiments/risk/risk_model_package.zip'
    log_step(f"Packaging all artifacts into {zip_path}...")
    
    files_to_zip = [
        # Reports
        'experiments/risk/TRAINING_REPORT_RISK.md',
        'experiments/risk/MODEL_CARD_RISK.md',
        'experiments/risk/dataset_analysis.md',
        # Config prediction outputs
        f'experiments/risk/clinical_demo_predictions_{m1_short}.json',
        f'experiments/risk/clinical_demo_predictions_{m2_short}.json',
        # Calibration logs
        f'experiments/risk/eval_predictions_{m1_short}.json',
        f'experiments/risk/eval_predictions_{m2_short}.json',
        f'experiments/risk/explainability_report_{m1_short}.md',
        f'experiments/risk/explainability_report_{m2_short}.md',
        # Curves
        f'experiments/risk/loss_curve_{m1_short}.png',
        f'experiments/risk/loss_curve_{m2_short}.png',
        f'experiments/risk/accuracy_curve_{m1_short}.png',
        f'experiments/risk/accuracy_curve_{m2_short}.png',
        f'experiments/risk/confusion_matrix_{m1_short}.png',
        f'experiments/risk/confusion_matrix_{m2_short}.png',
        # Weights (ONNX files)
        f'backend/models/risk_model_{m1_short}.onnx',
        f'backend/models/risk_model_{m1_short}_quant.onnx',
        f'backend/models/risk_model_{m2_short}.onnx',
        f'backend/models/risk_model_{m2_short}_quant.onnx',
    ]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for f in files_to_zip:
            if os.path.exists(f):
                # Preserve file tree layout relative to workspace
                zip_file.write(f)
            else:
                log_step(f"Warning: File not found for packaging: {f}")
                
    log_step("Zipping completed successfully.")
    
    # Google Colab auto-download trigger
    IS_COLAB = os.path.exists('/content')
    if IS_COLAB:
        try:
            from google.colab import files
            log_step("Colab environment: Triggering automatic package download...")
            files.download(zip_path)
        except Exception as e:
            log_step(f"Auto-download failed: {e}. You can download the zip package manually from your Colab file explorer: {zip_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Fast verification run')
    args = parser.parse_args()
    
    m1_name = 'emilyalsentzer/Bio_ClinicalBERT'
    m2_name = 'distilbert-base-uncased'
    
    m1_short = m1_name.split('/')[-1]
    m2_short = m2_name.split('/')[-1]
    
    metrics1 = train_model(m1_name, smoke_test=args.smoke_test)
    metrics2 = train_model(m2_name, smoke_test=args.smoke_test)
    
    if args.smoke_test:
        log_step("Smoke test complete: Skipping reports compilation.")
        return
        
    # Write TRAINING_REPORT_RISK.md
    report_lines = [
        "# Phase 5 Risk Model Comparative Training Report",
        "\nThis report summarizes the comparative results of fine-tuning two candidate architectures for triage severity prediction.",
        "\n## Model Comparison Matrix",
        "\n| Metric | Bio_ClinicalBERT | DistilBERT (Baseline) |",
        "| :--- | :---: | :---: |",
        f"| **Top-1 Validation Accuracy** | {metrics1['accuracy']:.4f} | {metrics2['accuracy']:.4f} |",
        f"| **Macro F1 Score** | {metrics1['macro_f1']:.4f} | {metrics2['macro_f1']:.4f} |",
        f"| **Weighted F1 Score** | {metrics1['weighted_f1']:.4f} | {metrics2['weighted_f1']:.4f} |",
        f"| **Precision (Weighted)** | {metrics1['precision']:.4f} | {metrics2['precision']:.4f} |",
        f"| **Recall (Weighted)** | {metrics1['recall']:.4f} | {metrics2['recall']:.4f} |",
        f"| **Expected Calibration Error (ECE)** | {metrics1['ece']:.4f} | {metrics2['ece']:.4f} |",
        f"| **Training Duration (s)** | {metrics1['duration_s']:.2f} | {metrics2['duration_s']:.2f} |",
        "\n## Selection Analysis",
        "\nBased on the validation metrics, `emilyalsentzer/Bio_ClinicalBERT` is evaluated side-by-side against `distilbert-base-uncased`. "
        "The model cards and exported quantized weights are prepared for Consensus Engine integration."
    ]
    
    with open('experiments/risk/TRAINING_REPORT_RISK.md', 'w') as f:
        f.write("\n".join(report_lines))
        
    # Write MODEL_CARD_RISK.md
    card_lines = [
        "# Model Card: ClinicalBERT Triage Classifier",
        "\n## Model Details",
        f"- **Model Developer**: Google Deepmind / Antigravity AI Project Team",
        f"- **Model Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}",
        f"- **Model Type**: Clinical Language Transformer (Sequence Classification)",
        f"- **Base Weights**: `emilyalsentzer/Bio_ClinicalBERT`",
        f"- **Training Dataset**: NHAMCS Emergency Department Triage",
        f"- **Language**: English",
        "\n## Intended Use",
        "- **Primary Use Case**: Automatic prediction of clinical triage severity classification based on natural text chief complaints.",
        "- **Triage Levels**: 0: Low, 1: Medium, 2: High, 3: Critical.",
        "\n## Quantitative Metrics",
        f"- **Val Accuracy**: {metrics1['accuracy']:.4f}",
        f"- **Macro F1**: {metrics1['macro_f1']:.4f}",
        f"- **Weighted F1**: {metrics1['weighted_f1']:.4f}",
        f"- **Target ECE**: {metrics1['ece']:.4f}"
    ]
    with open('experiments/risk/MODEL_CARD_RISK.md', 'w') as f:
        f.write("\n".join(card_lines))
        
    # Create Zip
    create_zip_package(m1_short, m2_short)
    log_step("Risk model pipeline run complete.")

if __name__ == '__main__':
    main()
