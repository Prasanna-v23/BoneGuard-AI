"""
BoneGuard AI - DenseNet121 + Standard CBAM Training Pipeline
=============================================================
Step 2: Train ImageNet-pretrained DenseNet121 with Standard CBAM on BFRMX Training split,
evaluate on BFRMX Validation split with early stopping, and save best checkpoint.

Rules strictly adhered to:
- Standard CBAM (Channel Attention + Spatial Attention) placed after each DenseBlock.
- NO DG-CBAM, no disagreement maps, no custom fusion.
- Class weights calculated ONLY from BFRMX Training set ([1.24475, 0.83568]).
- FracAtlas is NEVER used for training, hyperparameter tuning, or model selection.
- Complete metric logging: Loss, Accuracy, Precision, Recall, F1, ROC-AUC.
- Automatic device detection (CUDA GPU / CPU fallback).
"""

import os
import sys
import time
import random
import hashlib
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Add parent directory to path to import model module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.densenet121_cbam import build_densenet121_cbam, get_model_parameter_counts


# ==============================================================================
# 1. REPRODUCIBILITY & SEED CONFIGURATION
# ==============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 2. DATASET DEFINITION & VERIFICATION
# ==============================================================================
class BoneGuardXRayDataset(Dataset):
    """
    Robust PyTorch Dataset for BoneGuard AI radiographs.
    Ensures safe 3-channel RGB conversion for DenseNet121 without modifying disk files.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"FRACTURE": 0, "NO_FRACTURE": 1}
        self.idx_to_class = {0: "FRACTURE", 1: "NO_FRACTURE"}

        for class_name in ["FRACTURE", "NO_FRACTURE"]:
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                raise FileNotFoundError(f"Class folder not found: {class_folder}")
            
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(class_folder, fname)
                    self.samples.append((fpath, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, os.path.basename(path)

    def get_class_counts(self):
        fracture_count = sum(1 for _, l in self.samples if l == 0)
        no_fracture_count = sum(1 for _, l in self.samples if l == 1)
        return {"FRACTURE": fracture_count, "NO_FRACTURE": no_fracture_count}


def get_data_transforms():
    """
    Returns data transforms adhering strictly to project specification:
    - 224x224 spatial dimensions
    - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Training: Conservative mild affine/rotation augmentation (identical to baseline)
    - Validation/Testing: Deterministic resize & normalization only (No random augmentation)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform


# ==============================================================================
# 3. LEAKAGE AUDIT
# ==============================================================================
def compute_file_md5(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def verify_zero_leakage(train_ds, val_ds, test_ds, fracatlas_ds):
    """
    Verifies that no identical image content (cryptographic MD5 hash) is shared between any partitions.
    """
    train_hashes = set(compute_file_md5(p) for p, _ in train_ds.samples)
    val_hashes = set(compute_file_md5(p) for p, _ in val_ds.samples)
    test_hashes = set(compute_file_md5(p) for p, _ in test_ds.samples)
    frac_hashes = set(compute_file_md5(p) for p, _ in fracatlas_ds.samples)

    tv_overlap = train_hashes.intersection(val_hashes)
    tt_overlap = train_hashes.intersection(test_hashes)
    vt_overlap = val_hashes.intersection(test_hashes)
    tf_overlap = train_hashes.intersection(frac_hashes)

    if tv_overlap or tt_overlap or vt_overlap or tf_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED!\n"
            f"Train/Val hash overlap: {len(tv_overlap)}\n"
            f"Train/Test hash overlap: {len(tt_overlap)}\n"
            f"Val/Test hash overlap: {len(vt_overlap)}\n"
            f"Train/FracAtlas hash overlap: {len(tf_overlap)}"
        )
    print("[LEAKAGE AUDIT PASSED] 0 cryptographic content hash overlaps across Train, Val, Test, and External FracAtlas.")


# ==============================================================================
# 4. TRAINING & EVALUATION HELPER FUNCTIONS
# ==============================================================================
def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculates Accuracy, Precision, Recall, F1, and ROC-AUC (using positive class FRACTURE).
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Class 0 is FRACTURE (positive class) -> prob is y_prob[:, 0]
    y_true_binary = (np.array(y_true) == 0).astype(int)
    fracture_prob = y_prob[:, 0]
    
    try:
        roc_auc = roc_auc_score(y_true_binary, fracture_prob)
    except Exception:
        roc_auc = float('nan')

    return {
        "accuracy": float(acc),
        "precision": float(prec_macro),
        "recall": float(rec_macro),
        "f1": float(f1_macro),
        "roc_auc": float(roc_auc)
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Check for NaN / Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN or Inf loss encountered during training step!")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics, all_labels, all_preds, all_probs


# ==============================================================================
# 5. MAIN TRAINING WORKFLOW
# ==============================================================================
def run_training(
    data_split_dir: str,
    fracatlas_dir: str,
    output_dir: str,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 25,
    patience: int = 7,
    seed: int = 42
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
    
    print("=" * 70)
    print("BONEGUARD AI: DENSENET121 + STANDARD CBAM TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device} | GPU: {gpu_name} | CUDA: {cuda_ver}")
    print(f"PyTorch: {torch.__version__} | Random Seed: {seed}")
    print(f"Batch Size: {batch_size} | Learning Rate: {learning_rate} | Max Epochs: {max_epochs}")
    print(f"Output Directory: {output_dir}")

    # 2. Datasets & Transforms
    train_tf, eval_tf = get_data_transforms()
    train_dir = os.path.join(data_split_dir, "train")
    val_dir = os.path.join(data_split_dir, "validation")
    test_dir = os.path.join(data_split_dir, "test")

    train_ds = BoneGuardXRayDataset(train_dir, transform=train_tf)
    val_ds = BoneGuardXRayDataset(val_dir, transform=eval_tf)
    test_ds = BoneGuardXRayDataset(test_dir, transform=eval_tf)
    fracatlas_ds = BoneGuardXRayDataset(fracatlas_dir, transform=eval_tf)

    # Verify counts
    train_counts = train_ds.get_class_counts()
    val_counts = val_ds.get_class_counts()
    test_counts = test_ds.get_class_counts()
    frac_counts = fracatlas_ds.get_class_counts()

    print("\n--- AUTHORITATIVE DATASET PARTITION COUNTS ---")
    print(f"BFRMX Train:        FRACTURE={train_counts['FRACTURE']:4d}, NO_FRACTURE={train_counts['NO_FRACTURE']:4d} (Total: {len(train_ds)})")
    print(f"BFRMX Validation:   FRACTURE={val_counts['FRACTURE']:4d}, NO_FRACTURE={val_counts['NO_FRACTURE']:4d} (Total: {len(val_ds)})")
    print(f"BFRMX Test:         FRACTURE={test_counts['FRACTURE']:4d}, NO_FRACTURE={test_counts['NO_FRACTURE']:4d} (Total: {len(test_ds)})")
    print(f"FracAtlas External: FRACTURE={frac_counts['FRACTURE']:4d}, NO_FRACTURE={frac_counts['NO_FRACTURE']:4d} (Total: {len(fracatlas_ds)})")

    # Leakage check
    verify_zero_leakage(train_ds, val_ds, test_ds, fracatlas_ds)

    # 3. Class weights from BFRMX Train ONLY
    n_f = train_counts["FRACTURE"]
    n_nf = train_counts["NO_FRACTURE"]
    n_total = len(train_ds)
    w_f = n_total / (2.0 * n_f)
    w_nf = n_total / (2.0 * n_nf)
    class_weights = torch.tensor([w_f, w_nf], dtype=torch.float32).to(device)
    print(f"\nCalculated Training Class Weights (BFRMX Train ONLY):")
    print(f"  FRACTURE (idx=0):    {w_f:.5f}")
    print(f"  NO_FRACTURE (idx=1): {w_nf:.5f}")

    # 4. DataLoaders (num_workers=0 for high performance on Windows without IPC pipe overhead)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # 5. Model Instantiation
    model = build_densenet121_cbam(num_classes=2, pretrained=True).to(device)
    param_counts = get_model_parameter_counts(model)
    print(f"\nModel: DenseNet121 + Standard CBAM", flush=True)
    print(f"Total Parameters:     {param_counts['total_params']:,}", flush=True)
    print(f"Trainable Parameters: {param_counts['trainable_params']:,}", flush=True)
    print(f"Non-Trainable Params: {param_counts['non_trainable_params']:,}", flush=True)

    # Pre-training Verification & Sanity Check
    print("\n--- PRE-TRAINING VERIFICATION & SANITY CHECK ---", flush=True)
    print("1. Confirming CBAM modules present:")
    print(f"   cbam1 (ch=256):  {isinstance(model.cbam1, nn.Module)}")
    print(f"   cbam2 (ch=512):  {isinstance(model.cbam2, nn.Module)}")
    print(f"   cbam3 (ch=1024): {isinstance(model.cbam3, nn.Module)}")
    print(f"   cbam4 (ch=1024): {isinstance(model.cbam4, nn.Module)}")
    
    # Run dummy input [4, 3, 224, 224]
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    dummy_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long).to(device)
    dummy_out = model(dummy_input)
    print(f"2. Dummy input shape:  {list(dummy_input.shape)}")
    print(f"3. Dummy output shape: {list(dummy_out.shape)} (Expected: [4, 2])")
    assert list(dummy_out.shape) == [4, 2], f"Expected output shape [4, 2], got {list(dummy_out.shape)}"

    # Loss, backward, optimizer step verification
    criterion_test = nn.CrossEntropyLoss(weight=class_weights)
    dummy_loss = criterion_test(dummy_out, dummy_targets)
    print(f"4. Dummy batch loss:   {dummy_loss.item():.4f}")
    assert not (torch.isnan(dummy_loss) or torch.isinf(dummy_loss)), "Dummy loss is NaN or Inf!"
    
    dummy_loss.backward()
    opt_test = optim.AdamW(model.parameters(), lr=learning_rate)
    opt_test.step()
    print("5. Dummy backward pass and optimizer step: SUCCESS")
    print("Pre-training verification complete! Initializing fresh model for full training...\n", flush=True)

    # Reset model parameters to clean state
    model = build_densenet121_cbam(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # 6. Training Loop with Early Stopping
    history = []
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = -1
    best_metrics = {}
    epochs_no_improve = 0
    best_checkpoint_path = os.path.join(output_dir, "best_model.pth")

    start_time = time.time()
    print("=" * 70, flush=True)
    print(f"{'Epoch':^7}|{'Train Loss':^12}|{'Train Acc':^11}|{'Train F1':^10}|{'Val Loss':^10}|{'Val Acc':^9}|{'Val F1':^8}|{'Val AUC':^9}|{'Status':^8}", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(1, max_epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m, _, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_m["loss"])

        history_row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "train_roc_auc": train_m["roc_auc"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_roc_auc": val_m["roc_auc"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(history_row)

        status_str = ""
        # Model selection based on validation loss
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            best_metrics = val_m.copy()
            epochs_no_improve = 0
            status_str = "BEST *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_m,
                "train_metrics": train_m,
                "class_weights": class_weights.cpu().tolist(),
                "class_to_idx": train_ds.class_to_idx
            }, best_checkpoint_path)
        else:
            epochs_no_improve += 1
            status_str = f"no imp ({epochs_no_improve})"

        print(
            f"{epoch:^7d}|{train_m['loss']:^12.4f}|{train_m['accuracy']:^11.4f}|{train_m['f1']:^10.4f}|"
            f"{val_m['loss']:^10.4f}|{val_m['accuracy']:^9.4f}|{val_m['f1']:^8.4f}|{val_m['roc_auc']:^9.4f}|{status_str:^8}",
            flush=True
        )

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.")
            break

    total_training_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val F1: {best_val_f1:.4f}")

    # 7. Save Training History & Curves
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)

    # Save validation metrics at best epoch
    val_metrics_df = pd.DataFrame([best_metrics])
    val_metrics_df["best_epoch"] = best_epoch
    val_metrics_df.to_csv(os.path.join(output_dir, "validation_metrics.csv"), index=False)

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", color="#d62728", lw=2)
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("DenseNet121 + Standard CBAM - Loss vs Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Weighted CrossEntropy Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Plot Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy", color="#2ca02c", lw=2)
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("DenseNet121 + Standard CBAM - Accuracy vs Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # Plot Combined Training History Panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Loss
    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="Train", color="#1f77b4", lw=2)
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="Validation", color="#d62728", lw=2)
    axes[0, 0].axvline(best_epoch, color="gray", linestyle="--")
    axes[0, 0].set_title("Cross-Entropy Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy
    axes[0, 1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train", color="#2ca02c", lw=2)
    axes[0, 1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation", color="#ff7f0e", lw=2)
    axes[0, 1].axvline(best_epoch, color="gray", linestyle="--")
    axes[0, 1].set_title("Classification Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # F1-Score
    axes[1, 0].plot(history_df["epoch"], history_df["train_f1"], label="Train", color="#9467bd", lw=2)
    axes[1, 0].plot(history_df["epoch"], history_df["val_f1"], label="Validation", color="#8c564b", lw=2)
    axes[1, 0].axvline(best_epoch, color="gray", linestyle="--")
    axes[1, 0].set_title("Macro F1-Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1-Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # ROC-AUC
    axes[1, 1].plot(history_df["epoch"], history_df["train_roc_auc"], label="Train", color="#e377c2", lw=2)
    axes[1, 1].plot(history_df["epoch"], history_df["val_roc_auc"], label="Validation", color="#17becf", lw=2)
    axes[1, 1].axvline(best_epoch, color="gray", linestyle="--")
    axes[1, 1].set_title("ROC-AUC", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("DenseNet121 + Standard CBAM - Full Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    plt.close()

    # 8. Save Model Configuration & Summary
    config_text = f"""================================================================================
BONEGUARD AI: DENSENET121 + STANDARD CBAM CONFIGURATION & REPRODUCIBILITY LOG
================================================================================
Model Architecture:         DenseNet121 + Standard CBAM
Pretrained Backbone:        ImageNet-1K (DenseNet121_Weights.DEFAULT)
Attention Modules:          Standard CBAM (Channel Attention + Spatial Attention)
Attention Placement:        Output of each DenseBlock (denseblock 1, 2, 3, 4)
DG-CBAM Implemented:        NO (Strictly standard CBAM only)
Number of Classes:          2 (FRACTURE: 0, NO_FRACTURE: 1)
Input Dimensions:           224 x 224 x 3
Random Seed:                {seed}
Python Version:             {sys.version.split()[0]}
PyTorch Version:            {torch.__version__}
Device Selected:            {device}
GPU Name:                   {gpu_name}
CUDA Version:               {cuda_ver}
Batch Size:                 {batch_size}
Optimizer:                  AdamW (lr={learning_rate}, weight_decay={weight_decay})
LR Scheduler:               ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)
Loss Function:              nn.CrossEntropyLoss(weight=[{w_f:.5f}, {w_nf:.5f}])
Class Weight Source:        BFRMX Training Set ONLY (Fracture: {n_f}, No-Fracture: {n_nf})
Total Epochs Run:           {len(history)}
Best Validation Epoch:      {best_epoch}
Best Validation Loss:       {best_val_loss:.4f}
Best Validation F1-Score:   {best_val_f1:.4f}
Best Checkpoint Path:       {best_checkpoint_path}
================================================================================
"""
    with open(os.path.join(output_dir, "model_configuration.txt"), "w") as f:
        f.write(config_text)

    summary_text = f"""================================================================================
BONEGUARD AI: DENSENET121 + STANDARD CBAM MODEL ARCHITECTURE SUMMARY
================================================================================
Backbone:                   DenseNet-121 (DenseNet)
Feature Extractor:          121 layers with 4 DenseBlocks and Transition Layers
Attention Mechanism:        Standard CBAM (Woo et al., ECCV 2018)
CBAM Modules:
  - cbam1: Channel Attention (256 ch, r=16) + Spatial Attention (7x7 conv) -> 8,290 params
  - cbam2: Channel Attention (512 ch, r=16) + Spatial Attention (7x7 conv) -> 32,866 params
  - cbam3: Channel Attention (1024 ch, r=16) + Spatial Attention (7x7 conv) -> 131,170 params
  - cbam4: Channel Attention (1024 ch, r=16) + Spatial Attention (7x7 conv) -> 131,170 params
Total CBAM Parameters:      303,560
In-Features to Classifier:  1024
Classifier:                 nn.Linear(in_features=1024, out_features=2, bias=True)
Total Parameters:           {param_counts['total_params']:,}
Trainable Parameters:       {param_counts['trainable_params']:,}
Non-Trainable Parameters:   {param_counts['non_trainable_params']:,}
Output Classes:             2 classes -> [0: FRACTURE, 1: NO_FRACTURE]
Input Spatial Shape:        [Batch_Size, 3, 224, 224]
Final Logits Shape:         [Batch_Size, 2]
================================================================================
"""
    with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nTraining outputs, history, and configs saved to {output_dir}")
    return best_checkpoint_path, history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet121 + Standard CBAM for BoneGuard AI")
    parser.add_argument("--split_dir", type=str, default=r"D:\BoneGuard AI\04_Preprocessing\02_Split")
    parser.add_argument("--fracatlas_dir", type=str, default=r"D:\BoneGuard AI\04_Preprocessing\01_Cleaned\FracAtlas")
    parser.add_argument("--output_dir", type=str, default=r"D:\BoneGuard AI\06_Model_Results\DenseNet121_CBAM")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(
        data_split_dir=args.split_dir,
        fracatlas_dir=args.fracatlas_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed
    )
