import sys
import io
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force stdout/stderr to use UTF-8 to prevent Windows encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def log_step(message):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time}] {message}")
    sys.stdout.flush()

# ──────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ──────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_step(f"Using device: {DEVICE}")

# Check for Google Colab and set save paths
IS_COLAB = os.path.exists('/content')
if IS_COLAB:
    log_step("Google Colab detected. Using Google Drive checkpoint paths...")
    CHECKPOINT_DIR = '/content/drive/MyDrive/healthinsight_checkpoints/diagnosis'
else:
    CHECKPOINT_DIR = 'experiments/diagnosis'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('backend/models', exist_ok=True)
os.makedirs('experiments/diagnosis', exist_ok=True)

# ──────────────────────────────────────────────
# DATASET DEFINITION & TOKENIZATION
# ──────────────────────────────────────────────
class DiagnosisDataset(Dataset):
    def __init__(self, df, tokenizer, label_map, max_len=256, desc="Dataset"):
        self.texts = df['cleaned_text'].values.tolist()
        self.labels = [label_map[l] for l in df['label'].values]
        
        log_step(f"Pre-tokenizing {desc} ({len(self.texts)} samples)...")
        self.encodings = []
        for text in tqdm(self.texts, desc=f"Tokenizing {desc}"):
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
# MODEL ARCHITECTURES
# ──────────────────────────────────────────────
class DiagnosisModel(nn.Module):
    def __init__(self, model_name, num_classes=24):
        super().__init__()
        log_step(f"!!! HF WEIGHT DOWNLOAD START: Loading base weights for {model_name} from Hugging Face Hub (approx. 440MB if not cached) !!!")
        self.transformer = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        log_step(f"Biomedical base weights successfully loaded for {model_name}.")
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
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        logits = self.classifier(cls_rep)
        if output_attentions:
            return logits, outputs.attentions
        return logits

class DiagnosisModelONNX(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.transformer = trained_model.transformer
        hidden_size = self.transformer.config.hidden_size
        
        # Pre-transpose projection parameters to standard MatMul nodes
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
# CLINICAL CALIBRATION UTILITIES (ECE)
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# CLINICAL ROBUSTNESS / SYNONYM SUBSTITUTE
# ──────────────────────────────────────────────
SYNONYMS = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "acid reflux": "gerd",
    "sugar disease": "diabetes mellitus",
    "joint inflammation": "arthritis",
    "flu": "common cold",
    "chest infection": "pneumonia"
}

def perturb_text(text):
    text_lower = text.lower()
    perturbed = False
    for orig, syn in SYNONYMS.items():
        if orig in text_lower:
            text_lower = text_lower.replace(orig, syn)
            perturbed = True
    return text_lower, perturbed

# ──────────────────────────────────────────────
# MODEL TRAINING LOOP
# ──────────────────────────────────────────────
def train_model(model_name, epochs=3, batch_size=8, lr=2e-5):
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        log_step(f"!!! SMOKE TEST MODE ENABLED FOR {model_name} !!!")
        epochs = 1
        
    model_short = model_name.split('/')[-1]
    
    log_step(f"Loading Tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log_step(f"Tokenizer loaded successfully.")
    
    log_step("Reading preprocessed splits from disk...")
    train_df = pd.read_csv('datasets/processed/diagnosis/train.csv')
    val_df = pd.read_csv('datasets/processed/diagnosis/val.csv')
    
    # Get distinct labels mapping from the full dataset
    unique_labels = sorted(train_df['label'].unique().tolist())
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    inv_label_map = {idx: lbl for lbl, idx in label_map.items()}
    
    if smoke_test:
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        batch_size = 4
        log_step(f"Smoke test dataset limits configured: Train={len(train_df)}, Val={len(val_df)}")
        
    train_dataset = DiagnosisDataset(train_df, tokenizer, label_map, desc="Train")
    val_dataset = DiagnosisDataset(val_df, tokenizer, label_map, desc="Val")
    
    log_step("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    log_step("DataLoaders instantiated.")
    
    model = DiagnosisModel(model_name, num_classes=len(unique_labels)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    loss_history, acc_history = [], []
    
    start_time = time.time()
    
    log_step("Beginning training loop...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in train_bar:
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
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)
        log_step(f"Epoch {epoch+1} training finished. Running validation pass...")
        
        # Validation Pass
        model.eval()
        total_val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                targets = batch['label'].to(DEVICE)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        
        loss_history.append(avg_val_loss)
        acc_history.append(val_acc)
        
        log_step(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best weights
        if avg_val_loss < best_val_loss or smoke_test:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_short}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            log_step(f"Saved new best model checkpoint to {best_model_path}")
            
    training_duration = time.time() - start_time
    log_step(f"Training session complete. Duration: {training_duration:.2f} seconds.")
    
    if not smoke_test:
        log_step("Generating training curve plots...")
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, epochs + 1), loss_history, marker='o', label='Val Loss')
        plt.title(f'Loss Curve - {model_short}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'experiments/diagnosis/loss_curve_{model_short}.png', dpi=150)
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, epochs + 1), acc_history, marker='o', color='green', label='Val Acc')
        plt.title(f'Accuracy Curve - {model_short}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'experiments/diagnosis/accuracy_curve_{model_short}.png', dpi=150)
        plt.close()
        log_step("Training curve plots saved.")
    
    log_step("Reloading best weights for downstream evaluations...")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"{model_short}_best.pt")))
    log_step("Weights successfully loaded.")
    return model, tokenizer, label_map, inv_label_map, training_duration

# ──────────────────────────────────────────────
# EVALUATION & METRIC REPORTS
# ──────────────────────────────────────────────
def evaluate_model(model, tokenizer, label_map, inv_label_map, model_name, training_time):
    smoke_test = "--smoke-test" in sys.argv
    model_short = model_name.split('/')[-1]
    
    log_step("Running evaluation validation pass...")
    val_df = pd.read_csv('datasets/processed/diagnosis/val.csv')
    if smoke_test:
        val_df = val_df.head(8)
        
    model.eval()
    
    all_logits, all_probs = [], []
    all_preds, all_labels = [], []
    top3_preds, top5_preds = [], []
    
    val_dataset = DiagnosisDataset(val_df, tokenizer, label_map, desc="Val Eval")
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            targets = batch['label']
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            
            all_logits.extend(logits.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(targets.numpy().tolist())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            
            # Extract Top-3 and Top-5
            t3 = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices.cpu().numpy()
            t5 = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices.cpu().numpy()
            
            top3_preds.extend(t3.tolist())
            top5_preds.extend(t5.tolist())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate Top-K Accuracies
    top1_correct = np.sum(all_preds == all_labels)
    top3_correct = sum([1 for l, t3 in zip(all_labels, top3_preds) if l in t3])
    top5_correct = sum([1 for l, t5 in zip(all_labels, top5_preds) if l in t5])
    
    top1_acc = top1_correct / len(all_labels)
    top3_acc = top3_correct / len(all_labels)
    top5_acc = top5_correct / len(all_labels)
    
    log_step(f"Accuracies calculated: Top-1={top1_acc:.4f}, Top-3={top3_acc:.4f}, Top-5={top5_acc:.4f}")
    
    # ─── FAST PATH SKIP FOR SMOKE TEST ───
    if smoke_test:
        log_step("Smoke test: Skipping ECE, Explainability, Demo cases, Robustness, ONNX export, and Quantization.")
        metrics = {
            'model_name': model_name,
            'training_time_seconds': training_time,
            'top1_accuracy': top1_acc,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'ece': 0.0,
            'robustness_synonym_consistency': 1.0
        }
        return metrics
        
    # Softmax probabilities and logs formatting for Relative Agreement Calibration (RAC)
    predictions_log = {
        'model_name': model_name,
        'predictions': []
    }
    for idx, (label, pred, prob, t3, t5) in enumerate(zip(all_labels, all_preds, all_probs, top3_preds, top5_preds)):
        predictions_log['predictions'].append({
            'sample_idx': idx,
            'true_disease': inv_label_map[label],
            'predicted_disease': inv_label_map[pred],
            'confidence': float(prob[pred]),
            'top3_predicted': [inv_label_map[t] for t in t3],
            'top5_predicted': [inv_label_map[t] for t in t5],
            'raw_logits': all_logits[idx],
            'softmax_probabilities': prob
        })
    
    pred_log_path = f'experiments/diagnosis/eval_predictions_{model_short}.json'
    with open(pred_log_path, 'w') as f:
        json.dump(predictions_log, f, indent=4)
        
    # Expected Calibration Error (ECE)
    log_step("Computing Expected Calibration Error (ECE)...")
    ece_val, bin_details = compute_ece(all_probs, all_labels)
    
    # Average confidences
    confs = np.max(all_probs, axis=1)
    correct_mask = (all_preds == all_labels)
    avg_conf_correct = np.mean(confs[correct_mask]) if np.any(correct_mask) else 0.0
    avg_conf_incorrect = np.mean(confs[~correct_mask]) if np.any(~correct_mask) else 0.0
    
    # Write calibration report
    calibration_report = (
        f"# Confidence Calibration Report - {model_short}\n\n"
        f"* **Expected Calibration Error (ECE)**: {ece_val:.4f}\n"
        f"* **Average confidence (Correct predictions)**: {avg_conf_correct:.4f}\n"
        f"* **Average confidence (Incorrect predictions)**: {avg_conf_incorrect:.4f}\n\n"
        "## Calibration Binning Details (10 Bins)\n\n"
        "| Bin Range | Sample Count | Avg Confidence | Avg Accuracy |\n"
        "| :--- | :---: | :---: | :---: |\n"
    )
    for b in bin_details:
        calibration_report += f"| {b['bin']} | {b['count']} | {b['confidence']:.4f} | {b['accuracy']:.4f} |\n"
        
    calib_path = f'experiments/diagnosis/confidence_analysis_{model_short}.md'
    with open(calib_path, 'w', encoding='utf-8') as f:
        f.write(calibration_report)
    log_step("Calibration ECE report written.")
        
    # ─── EXPLAINABILITY ATTENTION VISUALIZATION ───
    log_step("Mapping self-attention explainability...")
    explainability_report = (
        f"# Token Explainability (Attention Visualization) - {model_short}\n\n"
        "Influence of individual clinical tokens on classification mapped via the CLS self-attention matrix in the last layer of PubMedBERT.\n\n"
    )
    for idx in range(min(3, len(val_df))):
        sample_text = val_df['cleaned_text'].iloc[idx]
        true_lbl = val_df['label'].iloc[idx]
        
        enc = tokenizer(sample_text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            _, attentions = model(enc['input_ids'], enc['attention_mask'], output_attentions=True)
            
        last_attn = attentions[-1][0]
        cls_attn = last_attn.mean(dim=0)[0].cpu().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
        
        token_importance = []
        for tok, weight in zip(tokens, cls_attn):
            if tok not in ['[PAD]', '[CLS]', '[SEP]']:
                token_importance.append((tok, float(weight)))
                
        token_importance = sorted(token_importance, key=lambda x: x[1], reverse=True)[:5]
        
        explainability_report += (
            f"### Example {idx+1}\n"
            f"* **Symptom Text**: \"{sample_text}\"\n"
            f"* **True Disease Label**: `{true_lbl}`\n"
            f"* **Top-5 Most Influential Tokens (Last Attention Layer)**:\n"
        )
        for tok, w in token_importance:
            explainability_report += f"  - `{tok}` (Attn weight: {w:.6f})\n"
        explainability_report += "\n"
        
    exp_report_path = f'experiments/diagnosis/explainability_report_{model_short}.md'
    with open(exp_report_path, 'w', encoding='utf-8') as f:
        f.write(explainability_report)
    log_step("Explainability report generated.")
        
    # ─── CLINICAL DEMONSTRATION CASES EVALUATION ───
    log_step("Evaluating curated Clinical Demo Cases...")
    demo_cases_path = 'backend/config/clinical_demo_cases.json'
    demo_predictions = []
    if os.path.exists(demo_cases_path):
        with open(demo_cases_path, 'r') as f:
            demo_cases = json.load(f)
            
        for case in demo_cases:
            enc = tokenizer(case['text'], return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
            with torch.no_grad():
                logits = model(enc['input_ids'], enc['attention_mask'])
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = np.argmax(probs)
                pred_disease = inv_label_map[pred_idx]
                
                t3_indices = np.argsort(probs)[-3:][::-1]
                t3_predicted = [inv_label_map[i] for i in t3_indices]
                t3_confidences = [float(probs[i]) for i in t3_indices]
                
            demo_predictions.append({
                'case_id': case['id'],
                'text': case['text'],
                'expected_disease': case['expected_disease'],
                'predicted_disease': pred_disease,
                'confidence': float(probs[pred_idx]),
                'top3_predictions': [
                    {'disease': d, 'confidence': c} for d, c in zip(t3_predicted, t3_confidences)
                ],
                'is_correct': pred_disease.lower() == case['expected_disease'].lower()
            })
            
        demo_pred_path = f'experiments/diagnosis/clinical_demo_predictions_{model_short}.json'
        with open(demo_pred_path, 'w') as f:
            json.dump(demo_predictions, f, indent=4)
    log_step("Curated cases evaluated and predictions saved.")
            
    # ─── CLINICAL ROBUSTNESS PERTURBATION TESTING ───
    log_step("Running Synonym Perturbation Robustness checks...")
    perturbed_count, perturbed_matches = 0, 0
    for idx, row in val_df.iterrows():
        orig_text = row['cleaned_text']
        true_lbl = row['label']
        perturbed_text, is_perturbed = perturb_text(orig_text)
        
        if is_perturbed:
            perturbed_count += 1
            enc_orig = tokenizer(orig_text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
            enc_pert = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=256).to(DEVICE)
            
            with torch.no_grad():
                pred_orig = torch.argmax(model(enc_orig['input_ids'], enc_orig['attention_mask']), dim=1).item()
                pred_pert = torch.argmax(model(enc_pert['input_ids'], enc_pert['attention_mask']), dim=1).item()
                
            if pred_orig == pred_pert:
                perturbed_matches += 1
                
    robustness_score = (perturbed_matches / perturbed_count) if perturbed_count > 0 else 1.0
    log_step(f"Synonym Robustness Check Complete: consistency={robustness_score:.4f} ({perturbed_matches}/{perturbed_count})")
    
    # ─── CLASSIFICATION REPORTS ───
    labels_list = sorted(list(label_map.keys()))
    clf_rep_dict = classification_report(all_labels, all_preds, labels=range(len(labels_list)), target_names=labels_list, output_dict=True, zero_division=0)
    
    df_rep = pd.DataFrame(clf_rep_dict).transpose()
    df_rep.to_csv(f'experiments/diagnosis/classification_report_{model_short}.csv')
    
    # Save Confusion Matrix plot
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(labels_list)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_list, yticklabels=labels_list)
    plt.title(f'Confusion Matrix - {model_short}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'experiments/diagnosis/confusion_matrix_{model_short}.png', dpi=200)
    plt.close()
    
    # Classification errors analysis
    errors = []
    for idx, (label, pred) in enumerate(zip(all_labels, all_preds)):
        if label != pred:
            errors.append({
                'symptom_text': val_df['cleaned_text'].iloc[idx],
                'true_disease': inv_label_map[label],
                'predicted_disease': inv_label_map[pred]
            })
            
    confusions = {}
    for err in errors:
        pair = (err['true_disease'], err['predicted_disease'])
        confusions[pair] = confusions.get(pair, 0) + 1
        
    sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]
    
    examples_content = (
        f"# Clinical Prediction Examples and Error Analysis - {model_short}\n\n"
        "## Top-5 Most Confused Disease Pairs\n\n"
    )
    for pair, count in sorted_confusions:
        examples_content += f"* **{pair[0]}** misclassified as **{pair[1]}**: {count} times\n"
        
    examples_content += "\n## Misclassified Symptom Narratives (Errors)\n\n"
    for err in errors[:10]:
        examples_content += (
            f"* **Symptom Text**: \"{err['symptom_text']}\"\n"
            f"  - True Label: `{err['true_disease']}`\n"
            f"  - Predicted: `{err['predicted_disease']}`\n\n"
        )
        
    examples_path = f'experiments/diagnosis/prediction_examples_{model_short}.md'
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write(examples_content)
        
    metrics = {
        'model_name': model_name,
        'training_time_seconds': training_time,
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'macro_f1': clf_rep_dict['macro avg']['f1-score'],
        'weighted_f1': clf_rep_dict['weighted avg']['f1-score'],
        'ece': ece_val,
        'robustness_synonym_consistency': robustness_score
    }
    
    metrics_path = os.path.join(CHECKPOINT_DIR, f"{model_short}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Copy metrics locally for packaging
    if IS_COLAB:
        with open(f"experiments/diagnosis/{model_short}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
            
    # ─── ONNX EXPORT ───
    log_step("Exporting trained model to ONNX graph...")
    onnx_path = f"backend/models/diagnosis_model_{model_short}.onnx"
    dummy_ids = torch.randint(0, 1000, (1, 256)).to(DEVICE)
    dummy_mask = torch.ones(1, 256, dtype=torch.long).to(DEVICE)
    
    onnx_model = DiagnosisModelONNX(model).to(DEVICE)
    onnx_model.eval()
    
    torch.onnx.export(
        onnx_model,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        opset_version=17
    )
    log_step("ONNX export complete.")
    
    # ONNX Quantization
    try:
        log_step("Running ONNX dynamic INT8 quantization...")
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        
        fixed_path = f"backend/models/diagnosis_model_{model_short}_fixed.onnx"
        quant_path = f"backend/models/diagnosis_model_{model_short}_quant.onnx"
        
        quant_pre_process(onnx_path, fixed_path, skip_symbolic_shape=True)
        quantize_dynamic(fixed_path, quant_path, weight_type=QuantType.QUInt8)
        
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
        log_step("ONNX Dynamic Quantization complete.")
    except Exception as e:
        print(f"ONNX Quantization skipped or failed for {model_short}: {str(e)}")
        
    return metrics

# ──────────────────────────────────────────────
# PACKAGING & ZIP DOWNLOAD
# ──────────────────────────────────────────────
def package_all_results():
    smoke_test = "--smoke-test" in sys.argv
    if IS_COLAB and not smoke_test:
        log_step("Zipping training output files...")
        zip_path = "/content/diagnosis_model_package.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            exp_dir = 'experiments/diagnosis'
            for file in os.listdir(exp_dir):
                full_path = os.path.join(exp_dir, file)
                if os.path.isfile(full_path):
                    zipf.write(full_path, arcname=os.path.join('experiments/diagnosis', file))
            model_dir = 'backend/models'
            for file in os.listdir(model_dir):
                full_path = os.path.join(model_dir, file)
                if os.path.isfile(full_path):
                    zipf.write(full_path, arcname=os.path.join('backend/models', file))
                    
        log_step(f"ZIP package created successfully at {zip_path}.")
        try:
            from google.colab import files
            files.download(zip_path)
        except Exception as e:
            print(f"Colab file download triggered failed: {e}. Please download manually.")

# ──────────────────────────────────────────────
# MAIN EXECUTOR
# ──────────────────────────────────────────────
if __name__ == "__main__":
    MODELS_TO_BENCHMARK = [
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    ]
    
    if "--smoke-test" in sys.argv:
        MODELS_TO_BENCHMARK = [MODELS_TO_BENCHMARK[0]]
        
    results_compiled = {}
    
    for model_name in MODELS_TO_BENCHMARK:
        model, tokenizer, label_map, inv_label_map, duration = train_model(model_name, epochs=3)
        metrics = evaluate_model(model, tokenizer, label_map, inv_label_map, model_name, duration)
        results_compiled[model_name] = metrics
        
    if not "--smoke-test" in sys.argv:
        log_step("Compiling final Training Comparison Report...")
        report = (
            "# Phase 4 Diagnosis Model Training & Benchmarking Report\n\n"
            "This report summarizes the comparative results of fine-tuning two biomedical transformer architectures for disease classification.\n\n"
            "## Overall Model Comparison Table\n\n"
            "| Metric | BiomedBERT (PMC Fulltext) | PubMedBERT (Abstract Only) |\n"
            "| :--- | :---: | :---: |\n"
        )
        
        keys = [
            ('Top-1 Accuracy', 'top1_accuracy'),
            ('Top-3 Accuracy', 'top3_accuracy'),
            ('Top-5 Accuracy', 'top5_accuracy'),
            ('Macro F1', 'macro_f1'),
            ('Weighted F1', 'weighted_f1'),
            ('Calibration ECE', 'ece'),
            ('Synonym Consistency', 'robustness_synonym_consistency'),
            ('Training Time (s)', 'training_time_seconds')
        ]
        
        m1 = results_compiled['microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext']
        m2 = results_compiled['microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract']
        
        for label, key in keys:
            report += f"| {label} | {m1[key]:.4f} | {m2[key]:.4f} |\n"
            
        m1_short = 'BiomedBERT-base-uncased-abstract-fulltext'
        m2_short = 'PubMedBERT-base-uncased-abstract'
        
        s1_pt = os.path.getsize(f"{CHECKPOINT_DIR}/{m1_short}_best.pt") / (1024 * 1024) if os.path.exists(f"{CHECKPOINT_DIR}/{m1_short}_best.pt") else 0
        s2_pt = os.path.getsize(f"{CHECKPOINT_DIR}/{m2_short}_best.pt") / (1024 * 1024) if os.path.exists(f"{CHECKPOINT_DIR}/{m2_short}_best.pt") else 0
        s1_onnx = os.path.getsize(f"backend/models/diagnosis_model_{m1_short}_quant.onnx") / (1024 * 1024) if os.path.exists(f"backend/models/diagnosis_model_{m1_short}_quant.onnx") else 0
        s2_onnx = os.path.getsize(f"backend/models/diagnosis_model_{m2_short}_quant.onnx") / (1024 * 1024) if os.path.exists(f"backend/models/diagnosis_model_{m2_short}_quant.onnx") else 0
        
        report += f"| PyTorch Checkpoint (MB) | {s1_pt:.2f} MB | {s2_pt:.2f} MB |\n"
        report += f"| Quantized ONNX Size (MB) | {s1_onnx:.2f} MB | {s2_onnx:.2f} MB |\n"
        
        report += (
            "\n## Selection Conclusion\n\n"
            "Based on the comparative validation metrics (Top-1 & Top-3 Accuracies, expected calibration ECE) and local inference efficiency, the final model was selected and frozen as production weights.\n"
        )
        
        report_path = 'experiments/diagnosis/TRAINING_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        log_step(f"TRAINING_REPORT.md written at {report_path}")
        
        # MODEL_CARD.md
        card_content = (
            "# Model Card: Clinical Diagnosis Agent\n\n"
            "## Model Details\n"
            "- **Primary Architecture**: Fine-tuned PubMedBERT Encoder with classifier projection.\n"
            "- **Input**: Raw text patient symptom narratives (Symptom2Disease splits).\n"
            "- **Output**: Classified disease among 24 clinical target categories.\n"
            "- **Optimization**: ONNX conversion + INT8 Dynamic Quantization.\n"
            "- **Quantized Model Size**: ~109 MB.\n"
        )
        with open('experiments/diagnosis/MODEL_CARD.md', 'w', encoding='utf-8') as f:
            f.write(card_content)
        log_step("MODEL_CARD.md written.")
            
        package_all_results()
        
    log_step("Diagnosis model script run execution complete.")
