import sys
import io
import os
# Add project root to sys.path to enable backend module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force stdout/stderr to use UTF-8 to prevent Windows encoding errors with emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ──────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Check for Google Colab and set save paths
IS_COLAB = os.path.exists('/content')
if IS_COLAB:
    print("Google Colab detected. Using Google Drive checkpoint paths...")
    CHECKPOINT_DIR = '/content/drive/MyDrive/healthinsight_checkpoints/intent'
else:
    CHECKPOINT_DIR = 'experiments/intent'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('backend/models', exist_ok=True)

# ──────────────────────────────────────────────
# LABEL MAPS
# ──────────────────────────────────────────────
DOMAIN_MAP = {"general": 0, "gastro": 1, "neuro": 2, "cardio": 3, "pulmo": 4}

# ──────────────────────────────────────────────
# DATASET IMPLEMENTATION (WITH PRE-TOKENIZATION)
# ──────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, desc="Dataset"):
        self.texts = df['cleaned_text'].values.tolist()
        self.domains = [DOMAIN_MAP[d] for d in df['medical_domain'].values]
        
        # Pre-tokenize to prevent blocking inside the DataLoader loop
        print(f"Pre-tokenizing {desc} ({len(self.texts)} samples)...")
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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.encodings[idx].copy()
        item['domain'] = torch.tensor(self.domains[idx], dtype=torch.long)
        return item

# ──────────────────────────────────────────────
# SINGLE-HEAD NETWORK MODEL
# ──────────────────────────────────────────────
class IntentModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        print(f"Downloading/Loading base model weights for {model_name}...")
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract CLS token representation & explicitly reshape to 2D
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        domain_logits = self.domain_head(cls_rep)
        return domain_logits

# ──────────────────────────────────────────────
# ONNX OPTIMIZED WRAPPER MODEL
# ──────────────────────────────────────────────
class IntentModelONNX(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.transformer = trained_model.transformer
        hidden_size = self.transformer.config.hidden_size
        
        # Domain head parameters (transposed weight matrices for MatMul nodes)
        self.dom_w1 = nn.Parameter(trained_model.domain_head[0].weight.t().clone())
        self.dom_b1 = nn.Parameter(trained_model.domain_head[0].bias.clone())
        self.dom_w2 = nn.Parameter(trained_model.domain_head[3].weight.t().clone())
        self.dom_b2 = nn.Parameter(trained_model.domain_head[3].bias.clone())
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs[0][:, 0, :].reshape(-1, self.transformer.config.hidden_size)
        
        # Domain head MatMul
        x_dom = torch.matmul(cls_rep, self.dom_w1) + self.dom_b1
        x_dom = torch.relu(x_dom)
        x_dom = torch.matmul(x_dom, self.dom_w2) + self.dom_b2
        
        return x_dom

# ──────────────────────────────────────────────
# MINOR CLASS OVERSAMPLING
# ──────────────────────────────────────────────
def balance_intent_dataset(df):
    """Mitigate class imbalance by oversampling pulmo and cardio cases."""
    df_minor_pulmo = df[df['medical_domain'] == 'pulmo']
    df_minor_cardio = df[df['medical_domain'] == 'cardio']
    
    # 10x oversample for pulmo, 5x for cardio
    pulmo_oversampled = pd.concat([df_minor_pulmo] * 10, ignore_index=True)
    cardio_oversampled = pd.concat([df_minor_cardio] * 5, ignore_index=True)
    
    balanced_df = pd.concat([df, pulmo_oversampled, cardio_oversampled], ignore_index=True)
    # Shuffle
    balanced_df = balanced_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return balanced_df

# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train_model(model_name='distilbert-base-uncased', epochs=3, batch_size=8, lr=2e-5):
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        print("\n!!! SMOKE TEST MODE ENABLED !!!")
        epochs = 1
        
    print(f"\n======================================")
    print(f"Training {model_name} (Single-Head)...")
    print(f"======================================")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = IntentModel(model_name).to(DEVICE)
    
    train_df = pd.read_csv('datasets/processed/intent/train.csv')
    val_df = pd.read_csv('datasets/processed/intent/val.csv')
    
    # Apply minor class oversampling
    train_df_balanced = balance_intent_dataset(train_df)
    
    if smoke_test:
        train_df_balanced = train_df_balanced.head(16)
        val_df = val_df.head(8)
        batch_size = 4
    
    # Calculate Class Weights for Loss Function
    domain_counts = train_df_balanced['medical_domain'].value_counts()
    class_weights = []
    for d in ["general", "gastro", "neuro", "cardio", "pulmo"]:
        count = domain_counts.get(d, 1)
        class_weights.append(len(train_df_balanced) / (5.0 * count))
    
    domain_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Calculated Domain Loss Weights: {class_weights}")
    
    train_dataset = IntentDataset(train_df_balanced, tokenizer, desc="Train Set")
    val_dataset = IntentDataset(val_df, tokenizer, desc="Val Set")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    
    criterion_domain = nn.CrossEntropyLoss(weight=domain_weights_tensor)
    
    best_val_loss = float('inf')
    early_stop_patience = 3
    patience_counter = 0
    
    start_time = time.time()
    
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{model_name.replace('/', '_')}_single_checkpoint.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_file) and not smoke_test:
        print(f"Loading checkpoint from {checkpoint_file} to resume training...")
        checkpoint = torch.load(checkpoint_file)
        # Handle state dict head size mismatch dynamically if moving from multi-head
        model_state = checkpoint['model_state']
        filtered_state = {}
        for k, v in model_state.items():
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(f"Skipping key {k} from checkpoint due to shape mismatch (head refactoring).")
        
        model.load_state_dict(filtered_state, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_loss']
        print(f"Resuming from Epoch {start_epoch}")
        
    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in train_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            target_domain = batch['domain'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            out_domain = model(input_ids, attention_mask)
            loss = criterion_domain(out_domain, target_domain)
            total_train_loss += loss.item()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            if step % 50 == 0 or smoke_test:
                print(f" Epoch {epoch+1} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation pass
        model.eval()
        total_val_loss = 0
        domain_preds, domain_trues = [], []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                target_domain = batch['domain'].to(DEVICE)
                
                out_domain = model(input_ids, attention_mask)
                loss = criterion_domain(out_domain, target_domain)
                total_val_loss += loss.item()
                
                preds = torch.argmax(out_domain, dim=1).cpu().numpy()
                domain_preds.extend(preds)
                domain_trues.extend(target_domain.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(domain_trues, domain_preds, average='macro', zero_division=0)
        
        print(f"\n--- Epoch {epoch+1} Summary - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Domain Val F1: {val_f1:.4f}")
        
        # Save epoch checkpoint to Google Drive/local
        if not smoke_test:
            checkpoint_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_val_loss
            }
            torch.save(checkpoint_state, checkpoint_file)
        
        if avg_val_loss < best_val_loss or smoke_test:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name.replace('/', '_')}_single_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f" --> Saved new best weights to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break
                
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name.replace('/', '_')}_single_best.pt")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    return model, tokenizer, duration

# ──────────────────────────────────────────────
# EVALUATION & EXPORT
# ──────────────────────────────────────────────
def evaluate_and_export(model, tokenizer, model_name, training_time):
    smoke_test = "--smoke-test" in sys.argv
    print(f"\nEvaluating and exporting {model_name}...")
    
    val_df = pd.read_csv('datasets/processed/intent/val.csv')
    if smoke_test:
        val_df = val_df.head(8)
        
    val_dataset = IntentDataset(val_df, tokenizer, desc="Val Set for Evaluation")
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    model.eval()
    domain_preds, domain_trues = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            out_domain = model(input_ids, attention_mask)
            domain_preds.extend(torch.argmax(out_domain, dim=1).cpu().numpy())
            domain_trues.extend(batch['domain'].numpy())
            
    # Generate reports directories
    os.makedirs('experiments/intent', exist_ok=True)
    
    # 1. Classification Reports text file
    print("Generating Classification Reports...")
    domain_names = ["general", "gastro", "neuro", "cardio", "pulmo"]
    domain_report = classification_report(domain_trues, domain_preds, labels=[0, 1, 2, 3, 4], target_names=domain_names, zero_division=0)
    
    report_content = (
        "==================================================\n"
        "INTENT CLASSIFIER MODULE (IPM) - EVALUATION REPORT\n"
        "==================================================\n\n"
        "--- CLINICAL MEDICAL DOMAIN PERFORMANCE ---\n"
        f"{domain_report}\n"
    )
    
    report_path = 'experiments/intent/classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved classification report to {report_path}")
    
    # 2. Confusion Matrix Plotting
    print("Generating Confusion Matrix plots...")
    cm_domain = confusion_matrix(domain_trues, domain_preds, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_domain, annot=True, fmt='d', cmap='Blues', xticklabels=domain_names, yticklabels=domain_names)
    plt.title('Medical Domain Classifier - Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('experiments/intent/confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved confusion matrix plot to experiments/intent/confusion_matrix.png")
    
    # Save validation metrics JSON
    domain_report_dict = classification_report(domain_trues, domain_preds, labels=[0, 1, 2, 3, 4], target_names=domain_names, output_dict=True, zero_division=0)
    metrics = {
        'model_name': model_name,
        'training_time_seconds': training_time,
        'device': str(DEVICE),
        'domain_accuracy': domain_report_dict['accuracy'],
        'domain_macro_f1': domain_report_dict['macro avg']['f1-score'],
        'domain_weighted_f1': domain_report_dict['weighted avg']['f1-score']
    }
    
    print("\n--- PERFORMANCE METRICS ---")
    print(json.dumps(metrics, indent=2))
    
    metrics_path = os.path.join(CHECKPOINT_DIR, f"{model_name.replace('/', '_')}_single_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Copy metrics to local experiments directory for packing
    if IS_COLAB:
        local_metrics_path = f"experiments/intent/distilbert-base-uncased_single_metrics.json"
        with open(local_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    # 3. ONNX Export with wrapper model (resolves Gemm transpose shape inference bugs)
    onnx_path = f"backend/models/intent_model_single.onnx"
    dummy_input = {
        'input_ids': torch.randint(0, 1000, (1, 512)).to(DEVICE),
        'attention_mask': torch.ones(1, 512, dtype=torch.long).to(DEVICE)
    }
    
    print("Instantiating ONNX-optimized wrapper model...")
    onnx_model = IntentModelONNX(model).to(DEVICE)
    onnx_model.eval()
    
    print(f"Exporting model to ONNX format at {onnx_path}...")
    torch.onnx.export(
        onnx_model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['domain_logits'],
        opset_version=17
    )
    print("ONNX Export complete.")
    
    # Quantize model using ONNX runtime and shape pre-processing
    try:
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        
        fixed_path = "backend/models/intent_model_single_fixed.onnx"
        quant_path = f"backend/models/intent_model_single_quant.onnx"
        
        print("Running ONNX shape pre-processing...")
        quant_pre_process(
            input_model_path=onnx_path,
            output_model_path=fixed_path,
            skip_symbolic_shape=True
        )
        
        print(f"Quantizing ONNX model to {quant_path}...")
        quantize_dynamic(
            fixed_path,
            quant_path,
            weight_type=QuantType.QUInt8
        )
        
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
            
        print("ONNX Quantization complete.")
    except Exception as e:
        print(f"ONNX Quantization skipped or failed: {str(e)}")

    # ──────────────────────────────────────────────
    # COLAB BROWSER DOWNLOADS PACKAGING
    # ──────────────────────────────────────────────
    if IS_COLAB and not smoke_test:
        print("\nPackaging checkpoints, metrics, and models for direct download...")
        zip_path = "/content/intent_model_single_package.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files from experiments/intent/
            exp_dir = 'experiments/intent'
            for file in os.listdir(exp_dir):
                full_path = os.path.join(exp_dir, file)
                if os.path.isfile(full_path):
                    zipf.write(full_path, arcname=os.path.join('experiments/intent', file))
            
            # Add files from backend/models/
            model_dir = 'backend/models'
            for file in os.listdir(model_dir):
                full_path = os.path.join(model_dir, file)
                if os.path.isfile(full_path):
                    zipf.write(full_path, arcname=os.path.join('backend/models', file))
                    
        print(f"ZIP package created successfully at {zip_path}!")
        
        # Trigger download in browser
        try:
            from google.colab import files
            print("Triggering browser download for package...")
            files.download(zip_path)
        except Exception as e:
            print(f"Browser download triggers failed: {str(e)}. Please download /content/intent_model_package.zip manually.")

# ──────────────────────────────────────────────
# MAIN RUNS
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer, duration = train_model('distilbert-base-uncased', epochs=3)
    evaluate_and_export(model, tokenizer, 'distilbert-base-uncased', duration)
