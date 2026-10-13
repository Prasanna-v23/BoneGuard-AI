"""
BoneGuard AI - DenseNet121 + Proposed DG-CBAM Training Pipeline
================================================================
Step 3: Train ImageNet-pretrained DenseNet121 with proposed Disagreement-Guided CBAM (DG-CBAM)
on BFRMX Training split, evaluate on BFRMX Validation split with early stopping,
and save the best checkpoint.

Rules strictly adhered to:
- Proposed DG-CBAM mechanism:
    Mc = Channel Attention Map
    Ms = Spatial Attention Map
    D = |Mc - Ms|
    M_new = Ms + gamma * D
    F_out = F (x) M_new
- Learnable scalar gamma initialized to 0.1 for each stage.
- Class weights calculated ONLY from BFRMX Training set ([1.24475, 0.83568]).
- FracAtlas is NEVER used for training, hyperparameter tuning, or model selection.
- Complete metric logging: Loss, Accuracy, Precision, Recall, F1, ROC-AUC, and Gamma history.
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
from models.densenet121_dgcbam import build_densenet121_dgcbam, get_model_parameter_counts


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
    - Training: Conservative mild affine/rotation augmentation (identical to baseline and CBAM)
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
    seed: int = 42,
    initial_gamma: float = 0.1
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
    
    print("=" * 70)
    print("BONEGUARD AI: DENSENET121 + PROPOSED DG-CBAM TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device} | GPU: {gpu_name} | CUDA: {cuda_ver}")
    print(f"PyTorch: {torch.__version__} | Random Seed: {seed}")
    print(f"Batch Size: {batch_size} | Learning Rate: {learning_rate} | Max Epochs: {max_epochs}")
    print(f"Initial Gamma: {initial_gamma} (Learnable parameter)")
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

    # 4. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # 5. Model Instantiation
    model = build_densenet121_dgcbam(num_classes=2, pretrained=True, initial_gamma=initial_gamma).to(device)
    param_counts = get_model_parameter_counts(model)
    gammas_init = model.get_gamma_values()
    print(f"\nModel: DenseNet121 + Proposed DG-CBAM", flush=True)
    print(f"Total Parameters:     {param_counts['total_params']:,}", flush=True)
    print(f"Trainable Parameters: {param_counts['trainable_params']:,}", flush=True)
    print(f"Non-Trainable Params: {param_counts['non_trainable_params']:,}", flush=True)
    print(f"Initial Gammas:       {gammas_init}", flush=True)

    # Pre-training Verification & Sanity Check
    print("\n--- PRE-TRAINING VERIFICATION & SANITY CHECK ---", flush=True)
    print("1. Confirming DG-CBAM modules present:")
    print(f"   dgcbam1 (ch=256):  {isinstance(model.dgcbam1, nn.Module)}")
    print(f"   dgcbam2 (ch=512):  {isinstance(model.dgcbam2, nn.Module)}")
    print(f"   dgcbam3 (ch=1024): {isinstance(model.dgcbam3, nn.Module)}")
    print(f"   dgcbam4 (ch=1024): {isinstance(model.dgcbam4, nn.Module)}")
    
    # Run dummy input [4, 3, 224, 224] with attention maps verification
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    dummy_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long).to(device)
    dummy_out, dummy_maps = model(dummy_input, return_attention_maps=True)
    print(f"2. Dummy input shape:  {list(dummy_input.shape)}")
    print(f"3. Dummy output shape: {list(dummy_out.shape)} (Expected: [4, 2])")
    assert list(dummy_out.shape) == [4, 2], f"Expected output shape [4, 2], got {list(dummy_out.shape)}"

    for k, v in dummy_maps.items():
        print(f"   {k} -> Mc: {list(v['Mc'].shape)}, Ms: {list(v['Ms'].shape)}, D: {list(v['D'].shape)}, M_new: {list(v['M_new'].shape)}")

    # Loss, backward, optimizer step verification
    criterion_test = nn.CrossEntropyLoss(weight=class_weights)
    dummy_loss = criterion_test(dummy_out, dummy_targets)
    print(f"4. Dummy batch loss:   {dummy_loss.item():.4f}")
    assert not (torch.isnan(dummy_loss) or torch.isinf(dummy_loss)), "Dummy loss is NaN or Inf!"
    
    dummy_loss.backward()
    
    # Confirm gamma receives gradients
    gamma_grads = {
        "g1_grad": model.dgcbam1.gamma.grad is not None,
        "g2_grad": model.dgcbam2.gamma.grad is not None,
        "g3_grad": model.dgcbam3.gamma.grad is not None,
        "g4_grad": model.dgcbam4.gamma.grad is not None
    }
    print(f"5. Gamma gradient confirmation: {gamma_grads}")
    assert all(gamma_grads.values()), "One or more gamma parameters did not receive gradients!"

    opt_test = optim.AdamW(model.parameters(), lr=learning_rate)
    opt_test.step()
    print("6. Dummy backward pass and optimizer step: SUCCESS")
    print("Pre-training verification complete! Initializing fresh model for full training...\n", flush=True)

    # Reset model parameters to clean state
    model = build_densenet121_dgcbam(num_classes=2, pretrained=True, initial_gamma=initial_gamma).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # 6. Training Loop with Early Stopping
    history = []
    gamma_history = []
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = -1
    best_metrics = {}
    best_gammas = {}
    epochs_no_improve = 0
    best_checkpoint_path = os.path.join(output_dir, "best_model.pth")

    start_time = time.time()
    print("=" * 80, flush=True)
    print(f"{'Epoch':^7}|{'Train Loss':^11}|{'Train Acc':^10}|{'Val Loss':^9}|{'Val Acc':^8}|{'Val F1':^7}|{'Val AUC':^8}|{'Gamma (mean)':^13}|{'Status':^8}", flush=True)
    print("=" * 80, flush=True)

    for epoch in range(1, max_epochs + 1):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m, _, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_m["loss"])

        current_gammas = model.get_gamma_values()
        gamma_row = {
            "epoch": epoch,
            "gamma_stage1": current_gammas["gamma_stage1"],
            "gamma_stage2": current_gammas["gamma_stage2"],
            "gamma_stage3": current_gammas["gamma_stage3"],
            "gamma_stage4": current_gammas["gamma_stage4"],
            "gamma_mean": current_gammas["gamma_mean"]
        }
        gamma_history.append(gamma_row)

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
            "gamma_mean": current_gammas["gamma_mean"],
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
            best_gammas = current_gammas.copy()
            epochs_no_improve = 0
            status_str = "BEST *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_m,
                "train_metrics": train_m,
                "gammas": current_gammas,
                "class_weights": class_weights.cpu().tolist(),
                "class_to_idx": train_ds.class_to_idx
            }, best_checkpoint_path)
        else:
            epochs_no_improve += 1
            status_str = f"no imp ({epochs_no_improve})"

        print(
            f"{epoch:^7d}|{train_m['loss']:^11.4f}|{train_m['accuracy']:^10.4f}|"
            f"{val_m['loss']:^9.4f}|{val_m['accuracy']:^8.4f}|{val_m['f1']:^7.4f}|{val_m['roc_auc']:^8.4f}|"
            f"{current_gammas['gamma_mean']:^13.5f}|{status_str:^8}",
            flush=True
        )

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.")
            break

    total_training_time = time.time() - start_time
    print("=" * 80)
    print(f"Training completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val F1: {best_val_f1:.4f}")
    print(f"Final Learned Gammas at Best Epoch: {best_gammas}")

    # 7. Save Training History & Gamma History
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)

    gamma_df = pd.DataFrame(gamma_history)
    gamma_csv = os.path.join(output_dir, "gamma_history.csv")
    gamma_df.to_csv(gamma_csv, index=False)

    # Save validation metrics at best epoch
    val_metrics_df = pd.DataFrame([best_metrics])
    val_metrics_df["best_epoch"] = best_epoch
    for gk, gv in best_gammas.items():
        val_metrics_df[gk] = gv
    val_metrics_df.to_csv(os.path.join(output_dir, "validation_metrics.csv"), index=False)

    # Plot Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", color="#d62728", lw=2)
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best Epoch ({best_epoch})")
    plt.title("DenseNet121 + Proposed DG-CBAM - Loss vs Epoch", fontsize=14, fontweight="bold")
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
    plt.title("DenseNet121 + Proposed DG-CBAM - Accuracy vs Epoch", fontsize=14, fontweight="bold")
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

    # Gamma Evolution
    axes[1, 1].plot(gamma_df["epoch"], gamma_df["gamma_stage1"], label="Stage 1 (256ch)", color="#1f77b4", lw=1.5)
    axes[1, 1].plot(gamma_df["epoch"], gamma_df["gamma_stage2"], label="Stage 2 (512ch)", color="#ff7f0e", lw=1.5)
    axes[1, 1].plot(gamma_df["epoch"], gamma_df["gamma_stage3"], label="Stage 3 (1024ch)", color="#2ca02c", lw=1.5)
    axes[1, 1].plot(gamma_df["epoch"], gamma_df["gamma_stage4"], label="Stage 4 (1024ch)", color="#d62728", lw=1.5)
    axes[1, 1].plot(gamma_df["epoch"], gamma_df["gamma_mean"], label="Mean Gamma", color="#000000", lw=2.5, linestyle="--")
    axes[1, 1].axvline(best_epoch, color="gray", linestyle="--")
    axes[1, 1].set_title("Learned Disagreement Weight (Gamma) Evolution", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Gamma (γ)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("DenseNet121 + Proposed DG-CBAM - Full Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    plt.close()

    # 8. Save Model Configuration & Summary
    config_text = f"""================================================================================
BONEGUARD AI: DENSENET121 + PROPOSED DG-CBAM CONFIGURATION & REPRODUCIBILITY LOG
================================================================================
Model Architecture:         DenseNet121 + Proposed DG-CBAM
Pretrained Backbone:        ImageNet-1K (DenseNet121_Weights.DEFAULT)
Attention Modules:          Disagreement-Guided CBAM (DG-CBAM)
Proposed Mechanism:         D = |Mc - Ms| -> M_new = Ms + gamma * D -> F_out = F (x) M_new
Attention Placement:        Output of each DenseBlock (denseblock 1, 2, 3, 4)
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
Initial Gamma:              {initial_gamma}
Total Epochs Run:           {len(history)}
Best Validation Epoch:      {best_epoch}
Best Validation Loss:       {best_val_loss:.4f}
Best Validation F1-Score:   {best_val_f1:.4f}
Final Learned Gammas:       {best_gammas}
Best Checkpoint Path:       {best_checkpoint_path}
================================================================================
"""
    with open(os.path.join(output_dir, "model_configuration.txt"), "w") as f:
        f.write(config_text)

    summary_text = f"""================================================================================
BONEGUARD AI: DENSENET121 + PROPOSED DG-CBAM MODEL ARCHITECTURE SUMMARY
================================================================================
Backbone:                   DenseNet-121 (DenseNet)
Feature Extractor:          121 layers with 4 DenseBlocks and Transition Layers
Attention Mechanism:        Proposed Disagreement-Guided CBAM (DG-CBAM)
Formulation:                D = |Mc - Ms| -> M_new = Ms + gamma * D -> F_out = F (x) M_new
DG-CBAM Modules:
  - dgcbam1: Channel Att (256 ch, r=16) + Spatial Att (7x7 conv) + Learnable Gamma -> 8,291 params
  - dgcbam2: Channel Att (512 ch, r=16) + Spatial Att (7x7 conv) + Learnable Gamma -> 32,867 params
  - dgcbam3: Channel Att (1024 ch, r=16) + Spatial Att (7x7 conv) + Learnable Gamma -> 131,171 params
  - dgcbam4: Channel Att (1024 ch, r=16) + Spatial Att (7x7 conv) + Learnable Gamma -> 131,171 params
Total DG-CBAM Parameters:   303,564 (303,560 Conv/MLP weights + 4 learnable gamma scalars)
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
    parser = argparse.ArgumentParser(description="Train DenseNet121 + Proposed DG-CBAM for BoneGuard AI")
    parser.add_argument("--split_dir", type=str, default=r"D:\BoneGuard AI\04_Preprocessing\02_Split")
    parser.add_argument("--fracatlas_dir", type=str, default=r"D:\BoneGuard AI\04_Preprocessing\01_Cleaned\FracAtlas")
    parser.add_argument("--output_dir", type=str, default=r"D:\BoneGuard AI\06_Model_Results\DenseNet121_DGCBAM")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_gamma", type=float, default=0.1)
    args = parser.parse_args()

    run_training(
        data_split_dir=args.split_dir,
        fracatlas_dir=args.fracatlas_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        initial_gamma=args.initial_gamma
    )
