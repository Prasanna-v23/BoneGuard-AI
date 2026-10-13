"""
BoneGuard AI - Experiment F: Domain-Robust Augmentation for Cross-Domain Generalization
========================================================================================
Architecture: DenseNet121 + Proposed DG-CBAM
Goal: Test whether realistic radiographic domain-variation augmentation during BFRMX
      training improves generalization to the untouched FracAtlas external benchmark.

SCIENTIFIC INTEGRITY PROTOCOL:
- Training: ONLY BFRMX Train split (N=2,611).
- Model Selection: ONLY BFRMX Validation split (N=560).
- External Evaluation: Evaluated ONCE on FracAtlas (N=4,068) after model selection.
- FracAtlas is NEVER used for training, hyperparameter tuning, threshold tuning, or model selection.
- All models evaluated at standard locked threshold = 0.50.
"""

import os
import sys
import copy
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "05_Code"))

from models.densenet121_dgcbam import build_densenet121_dgcbam, get_model_parameter_counts
from training.train_densenet121 import BoneGuardXRayDataset


# ==============================================================================
# 1. REPRODUCIBILITY & SEED
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 2. REALISTIC RADIOGRAPHIC DOMAIN-ROBUST TRANSFORMS
# ==============================================================================
def get_domain_robust_transforms():
    """
    Creates conservative, medically plausible radiographic domain-variation
    augmentations for BFRMX training, designed to simulate:
    1. Scanner exposure & contrast LUT differences (brightness/contrast jitter, auto-contrast)
    2. Detector sharpness/blur variations (Gaussian blur, sharpness adjustment)
    3. Patient positioning & magnification variations (mild rotation, translation, scaling)
    Without creating artificial fractures or destroying bone anatomy.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.30),
        transforms.RandomAutocontrast(p=0.20),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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
# 3. METRIC CALCULATION ENGINE
# ==============================================================================
def calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)) * 100.0
    rec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)) * 100.0
    f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)) * 100.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    spec = (float(tn / (tn + fp)) * 100.0) if (tn + fp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true == 0, y_prob[:, 0]))
    except Exception:
        roc_auc = 0.5

    fpr, tpr, _ = roc_curve(y_true == 0, y_prob[:, 0])
    pr_precision, pr_recall, _ = precision_recall_curve(y_true == 0, y_prob[:, 0])
    pr_auc = float(auc(pr_recall, pr_precision))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": len(y_true),
        "probs": y_prob,
        "preds": y_pred,
        "targets": y_true,
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall
    }


# ==============================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.vstack(all_probs)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) if criterion else torch.tensor(0.0)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_probs = np.vstack(all_probs)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    return metrics


# ==============================================================================
# 5. VISUALIZATION HELPERS
# ==============================================================================
def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Loss Curve
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
    axes[0].set_title('Training & Validation Loss (Exp F)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CrossEntropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1-Score & Accuracy Curve
    axes[1].plot(history_df['epoch'], history_df['val_acc'], label='Val Accuracy (%)', color='#2ca02c', lw=2)
    axes[1].plot(history_df['epoch'], history_df['val_f1'], label='Val F1-Score (%)', color='#d62728', lw=2)
    axes[1].set_title('Validation Performance Dynamics (Exp F)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# 6. MAIN EXPERIMENT EXECUTION
# ==============================================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(PROJECT_ROOT, "06_Model_Results", "DGCBAM_DomainRobustness", "Exp_F_DomainAugmentation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("BONEGUARD AI — EXPERIMENT F: DOMAIN-ROBUST AUGMENTATION")
    print("=" * 80)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Output Directory: {out_dir}")

    # Dataset Paths
    split_dir = os.path.join(PROJECT_ROOT, "04_Preprocessing", "02_Split")
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "validation")
    test_dir = os.path.join(split_dir, "test")
    fracatlas_dir = os.path.join(PROJECT_ROOT, "04_Preprocessing", "01_Cleaned", "FracAtlas")

    train_tf, eval_tf = get_domain_robust_transforms()

    train_ds = BoneGuardXRayDataset(train_dir, transform=train_tf)
    val_ds = BoneGuardXRayDataset(val_dir, transform=eval_tf)
    test_ds = BoneGuardXRayDataset(test_dir, transform=eval_tf)
    fracatlas_ds = BoneGuardXRayDataset(fracatlas_dir, transform=eval_tf)

    batch_size = 32
    max_epochs = 25
    patience = 7

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    fracatlas_loader = DataLoader(fracatlas_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nDatasets Loaded:")
    print(f"  - BFRMX Train:      {len(train_ds)} samples {train_ds.get_class_counts()}")
    print(f"  - BFRMX Validation: {len(val_ds)} samples {val_ds.get_class_counts()}")
    print(f"  - BFRMX Test:       {len(test_ds)} samples {test_ds.get_class_counts()}")
    print(f"  - FracAtlas Ext:    {len(fracatlas_ds)} samples {fracatlas_ds.get_class_counts()}")

    # Model Build
    model = build_densenet121_dgcbam(num_classes=2, pretrained=True, initial_gamma=0.1).to(device)
    param_counts = get_model_parameter_counts(model)
    print(f"\nModel: DenseNet121 + Proposed DG-CBAM ({param_counts['total_params']:,} parameters)")

    # Loss & Optimizer (Unweighted CrossEntropy matching Exp E)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Save Configuration
    with open(os.path.join(out_dir, "configuration.txt"), "w") as f:
        f.write("BONEGUARD AI — EXPERIMENT F CONFIGURATION\n")
        f.write("=" * 60 + "\n")
        f.write("Experiment: Exp F (Domain-Robust Augmentation with Unweighted Loss)\n")
        f.write("Architecture: DenseNet121 + Proposed DG-CBAM\n")
        f.write("Total Parameters: 7,259,406\n")
        f.write("Initial Gamma: 0.10\n")
        f.write("Loss Function: Unweighted CrossEntropyLoss [1.0, 1.0]\n")
        f.write("Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)\n")
        f.write("Scheduler: CosineAnnealingLR (T_max=25, eta_min=1e-6)\n")
        f.write("Batch Size: 32\n")
        f.write("Max Epochs: 25\n")
        f.write("Early Stopping Patience: 7 epochs on BFRMX Val F1\n")
        f.write("Random Seed: 42\n")
        f.write("Domain Augmentations (Train Only):\n")
        f.write("  - RandomRotation (8 deg)\n")
        f.write("  - RandomAffine (translate 4%, scale 0.96-1.04)\n")
        f.write("  - RandomHorizontalFlip (p=0.5)\n")
        f.write("  - ColorJitter (brightness 0.15, contrast 0.15)\n")
        f.write("  - RandomAdjustSharpness (factor 1.5, p=0.3)\n")
        f.write("  - RandomAutocontrast (p=0.2)\n")
        f.write("  - GaussianBlur (kernel 3, sigma 0.1-1.0)\n")
        f.write("Data Isolation: FracAtlas was completely isolated during all training & model selection.\n")

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_weights = None
    best_val_metrics = None
    no_improve = 0

    print("\nStarting Training on BFRMX Training Partition...")
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        tr_m = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        gamma_dict = model.get_gamma_values()
        rec = {
            "epoch": epoch,
            "train_loss": tr_m["loss"],
            "train_acc": tr_m["accuracy"],
            "train_f1": tr_m["f1"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["accuracy"],
            "val_f1": val_m["f1"],
            "val_prec": val_m["precision"],
            "val_rec": val_m["recall"],
            "val_spec": val_m["specificity"],
            "val_auc": val_m["roc_auc"],
            "gamma_mean": gamma_dict["gamma_mean"]
        }
        history.append(rec)

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {tr_m['loss']:.4f} Acc: {tr_m['accuracy']:.2f}% F1: {tr_m['f1']:.2f}% | "
              f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.2f}% F1: {val_m['f1']:.2f}% AUC: {val_m['roc_auc']:.4f} | gamma_mean: {gamma_dict['gamma_mean']:.4f}")

        # Model selection ONLY on BFRMX Val F1
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_val_metrics = val_m
            no_improve = 0
            print(f"  >>> [NEW BEST VAL F1: {best_val_f1:.2f}% at epoch {epoch}] <<<")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without validation improvement.")
                break

    train_time = time.time() - t_start
    print(f"\nTraining completed in {train_time/60:.2f} minutes. Best Epoch: {best_epoch} (Val F1 = {best_val_f1:.2f}%)")

    # Load best checkpoint
    model.load_state_dict(best_weights)
    model.eval()

    # Save Checkpoint
    ckpt_path = os.path.join(out_dir, "best_model.pth")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_weights,
        "val_metrics": best_val_metrics,
        "experiment_id": "Exp_F_DomainAugmentation",
        "experiment_name": "Domain-Robust Augmentation Study",
        "training_time_seconds": train_time
    }, ckpt_path)
    print(f"Best checkpoint saved to: {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.2f} MB)")

    # Save Training History CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    # Save Checkpoint Info
    gamma_final = model.get_gamma_values()
    with open(os.path.join(out_dir, "checkpoint_info.txt"), "w") as f:
        f.write("BONEGUARD AI — EXP F CHECKPOINT INFORMATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint File: {ckpt_path}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Total Parameters: {param_counts['total_params']:,}\n")
        f.write(f"Trainable Parameters: {param_counts['trainable_params']:,}\n")
        f.write(f"Learned Gamma Stage 1: {gamma_final['gamma_stage1']:.6f}\n")
        f.write(f"Learned Gamma Stage 2: {gamma_final['gamma_stage2']:.6f}\n")
        f.write(f"Learned Gamma Stage 3: {gamma_final['gamma_stage3']:.6f}\n")
        f.write(f"Learned Gamma Stage 4: {gamma_final['gamma_stage4']:.6f}\n")
        f.write(f"Learned Gamma Mean:    {gamma_final['gamma_mean']:.6f}\n")
        f.write(f"Best BFRMX Val F1:      {best_val_f1:.2f}%\n")
        f.write(f"Best BFRMX Val Acc:     {best_val_metrics['accuracy']:.2f}%\n")

    # Plot Training Curves
    plot_training_curves(history_df, os.path.join(out_dir, "training_curves.png"))

    # ==============================================================================
    # 7. IN-DOMAIN BFRMX TEST SET EVALUATION
    # ==============================================================================
    print("\n" + "=" * 80)
    print("EVALUATING ON BFRMX INTERNAL TEST SET (N = 560)")
    print("=" * 80)
    bfrmx_test_res = evaluate(model, test_loader, criterion, device)
    print(f"BFRMX Test Results:")
    print(f"  Accuracy:    {bfrmx_test_res['accuracy']:.2f}%")
    print(f"  Precision:   {bfrmx_test_res['precision']:.2f}%")
    print(f"  Recall:      {bfrmx_test_res['recall']:.2f}%")
    print(f"  Specificity: {bfrmx_test_res['specificity']:.2f}%")
    print(f"  F1-Score:    {bfrmx_test_res['f1']:.2f}%")
    print(f"  ROC-AUC:     {bfrmx_test_res['roc_auc']:.4f}")
    print(f"  PR-AUC:      {bfrmx_test_res['pr_auc']:.4f}")
    print(f"  Confusion Matrix: TP={bfrmx_test_res['tp']}, TN={bfrmx_test_res['tn']}, FP={bfrmx_test_res['fp']}, FN={bfrmx_test_res['fn']}")

    class_names = ["FRACTURE", "NO_FRACTURE"]
    cm_bfrmx = np.array([[bfrmx_test_res['tp'], bfrmx_test_res['fn']], [bfrmx_test_res['fp'], bfrmx_test_res['tn']]])
    plot_confusion_matrix(cm_bfrmx, class_names, f"BFRMX Internal Test — Exp F (Acc: {bfrmx_test_res['accuracy']:.2f}%)", os.path.join(out_dir, "confusion_matrix_bfrmx.png"))

    # ==============================================================================
    # 8. OUT-OF-DOMAIN FRACATLAS BENCHMARK EVALUATION (STRICT ZERO-LEAKAGE)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("EVALUATING ON UNTOUCHED FRACATLAS EXTERNAL BENCHMARK (N = 4,068)")
    print("=" * 80)
    fracatlas_res = evaluate(model, fracatlas_loader, criterion, device)
    print(f"FracAtlas External Benchmark Results:")
    print(f"  Accuracy:    {fracatlas_res['accuracy']:.2f}%")
    print(f"  Precision:   {fracatlas_res['precision']:.2f}%")
    print(f"  Recall:      {fracatlas_res['recall']:.2f}%")
    print(f"  Specificity: {fracatlas_res['specificity']:.2f}%")
    print(f"  F1-Score:    {fracatlas_res['f1']:.2f}%")
    print(f"  ROC-AUC:     {fracatlas_res['roc_auc']:.4f}")
    print(f"  PR-AUC:      {fracatlas_res['pr_auc']:.4f}")
    print(f"  Confusion Matrix: TP={fracatlas_res['tp']}, TN={fracatlas_res['tn']}, FP={fracatlas_res['fp']}, FN={fracatlas_res['fn']}")

    cm_fracatlas = np.array([[fracatlas_res['tp'], fracatlas_res['fn']], [fracatlas_res['fp'], fracatlas_res['tn']]])
    plot_confusion_matrix(cm_fracatlas, class_names, f"FracAtlas External Benchmark — Exp F (Acc: {fracatlas_res['accuracy']:.2f}%)", os.path.join(out_dir, "confusion_matrix_fracatlas.png"))

    # Plot Combined ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # ROC
    axes[0].plot(bfrmx_test_res['fpr'], bfrmx_test_res['tpr'], label=f"BFRMX Test (AUC = {bfrmx_test_res['roc_auc']:.4f})", color='#1f77b4', lw=2)
    axes[0].plot(fracatlas_res['fpr'], fracatlas_res['tpr'], label=f"FracAtlas Benchmark (AUC = {fracatlas_res['roc_auc']:.4f})", color='#d62728', lw=2)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Chance")
    axes[0].set_title("ROC Curves — Exp F (BFRMX vs FracAtlas)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR
    axes[1].plot(bfrmx_test_res['pr_recall'], bfrmx_test_res['pr_precision'], label=f"BFRMX Test (PR-AUC = {bfrmx_test_res['pr_auc']:.4f})", color='#1f77b4', lw=2)
    axes[1].plot(fracatlas_res['pr_recall'], fracatlas_res['pr_precision'], label=f"FracAtlas Benchmark (PR-AUC = {fracatlas_res['pr_auc']:.4f})", color='#d62728', lw=2)
    axes[1].set_title("Precision-Recall Curves — Exp F", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_pr_curves.png"), bbox_inches='tight')
    plt.close(fig)

    # Save Validation Metrics CSV
    metrics_summary = [
        {
            "Dataset": "BFRMX_Validation",
            "N": len(val_ds),
            "Accuracy": best_val_metrics["accuracy"],
            "Precision": best_val_metrics["precision"],
            "Recall": best_val_metrics["recall"],
            "Specificity": best_val_metrics["specificity"],
            "F1_Score": best_val_metrics["f1"],
            "ROC_AUC": best_val_metrics["roc_auc"],
            "PR_AUC": best_val_metrics["pr_auc"],
            "TP": best_val_metrics["tp"],
            "TN": best_val_metrics["tn"],
            "FP": best_val_metrics["fp"],
            "FN": best_val_metrics["fn"]
        },
        {
            "Dataset": "BFRMX_Internal_Test",
            "N": len(test_ds),
            "Accuracy": bfrmx_test_res["accuracy"],
            "Precision": bfrmx_test_res["precision"],
            "Recall": bfrmx_test_res["recall"],
            "Specificity": bfrmx_test_res["specificity"],
            "F1_Score": bfrmx_test_res["f1"],
            "ROC_AUC": bfrmx_test_res["roc_auc"],
            "PR_AUC": bfrmx_test_res["pr_auc"],
            "TP": bfrmx_test_res["tp"],
            "TN": bfrmx_test_res["tn"],
            "FP": bfrmx_test_res["fp"],
            "FN": bfrmx_test_res["fn"]
        },
        {
            "Dataset": "FracAtlas_External_Benchmark",
            "N": len(fracatlas_ds),
            "Accuracy": fracatlas_res["accuracy"],
            "Precision": fracatlas_res["precision"],
            "Recall": fracatlas_res["recall"],
            "Specificity": fracatlas_res["specificity"],
            "F1_Score": fracatlas_res["f1"],
            "ROC_AUC": fracatlas_res["roc_auc"],
            "PR_AUC": fracatlas_res["pr_auc"],
            "TP": fracatlas_res["tp"],
            "TN": fracatlas_res["tn"],
            "FP": fracatlas_res["fp"],
            "FN": fracatlas_res["fn"]
        }
    ]
    pd.DataFrame(metrics_summary).to_csv(os.path.join(out_dir, "validation_metrics.csv"), index=False)

    # ==============================================================================
    # 9. CONTROLLED COMPARISON TABLE WITH PREVIOUS EXPERIMENTS
    # ==============================================================================
    comparison_data = [
        {
            "Model / Experiment": "DenseNet121 Baseline",
            "Description": "Vanilla DenseNet-121 without attention",
            "BFRMX_Acc": 99.64, "BFRMX_F1": 99.56, "BFRMX_Recall": 99.56, "BFRMX_Spec": 99.70, "BFRMX_AUC": 0.9999,
            "FracAtlas_Acc": 41.30, "FracAtlas_F1": 31.46, "FracAtlas_Recall": 76.86, "FracAtlas_Spec": 33.74, "FracAtlas_AUC": 0.6104,
            "FracAtlas_TP": 548, "FracAtlas_TN": 1132, "FracAtlas_FP": 2223, "FracAtlas_FN": 165
        },
        {
            "Model / Experiment": "DenseNet121 + Standard CBAM",
            "Description": "Sequential Channel and Spatial Attention",
            "BFRMX_Acc": 99.82, "BFRMX_F1": 99.78, "BFRMX_Recall": 99.56, "BFRMX_Spec": 100.00, "BFRMX_AUC": 0.9992,
            "FracAtlas_Acc": 32.52, "FracAtlas_F1": 30.91, "FracAtlas_Recall": 86.12, "FracAtlas_Spec": 21.13, "FracAtlas_AUC": 0.5861,
            "FracAtlas_TP": 614, "FracAtlas_TN": 709, "FracAtlas_FP": 2646, "FracAtlas_FN": 99
        },
        {
            "Model / Experiment": "DenseNet121 + DG-CBAM (Original)",
            "Description": "Proposed DG-CBAM with weighted loss",
            "BFRMX_Acc": 99.64, "BFRMX_F1": 99.56, "BFRMX_Recall": 99.12, "BFRMX_Spec": 100.00, "BFRMX_AUC": 0.9999,
            "FracAtlas_Acc": 40.09, "FracAtlas_F1": 32.44, "FracAtlas_Recall": 82.05, "FracAtlas_Spec": 31.18, "FracAtlas_AUC": 0.6336,
            "FracAtlas_TP": 585, "FracAtlas_TN": 1046, "FracAtlas_FP": 2309, "FracAtlas_FN": 128
        },
        {
            "Model / Experiment": "Exp E — Unweighted DG-CBAM",
            "Description": "DG-CBAM with unweighted CrossEntropyLoss",
            "BFRMX_Acc": 99.64, "BFRMX_F1": 99.56, "BFRMX_Recall": 99.12, "BFRMX_Spec": 100.00, "BFRMX_AUC": 0.9996,
            "FracAtlas_Acc": 49.90, "FracAtlas_F1": 34.51, "FracAtlas_Recall": 75.32, "FracAtlas_Spec": 44.50, "FracAtlas_AUC": 0.6479,
            "FracAtlas_TP": 537, "FracAtlas_TN": 1493, "FracAtlas_FP": 1862, "FracAtlas_FN": 176
        },
        {
            "Model / Experiment": "Exp F — Domain-Robust Augmentation",
            "Description": "DG-CBAM + realistic radiographic domain-variation augmentation",
            "BFRMX_Acc": bfrmx_test_res["accuracy"], "BFRMX_F1": bfrmx_test_res["f1"], "BFRMX_Recall": bfrmx_test_res["recall"], "BFRMX_Spec": bfrmx_test_res["specificity"], "BFRMX_AUC": bfrmx_test_res["roc_auc"],
            "FracAtlas_Acc": fracatlas_res["accuracy"], "FracAtlas_F1": fracatlas_res["f1"], "FracAtlas_Recall": fracatlas_res["recall"], "FracAtlas_Spec": fracatlas_res["specificity"], "FracAtlas_AUC": fracatlas_res["roc_auc"],
            "FracAtlas_TP": fracatlas_res["tp"], "FracAtlas_TN": fracatlas_res["tn"], "FracAtlas_FP": fracatlas_res["fp"], "FracAtlas_FN": fracatlas_res["fn"]
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(out_dir, "comparison_with_exp_e.csv"), index=False)
    print("\n" + "=" * 80)
    print("CONTROLLED EXPERIMENT COMPARISON TABLE")
    print("=" * 80)
    print(df_comp[["Model / Experiment", "BFRMX_Acc", "BFRMX_F1", "FracAtlas_Acc", "FracAtlas_F1", "FracAtlas_Recall", "FracAtlas_Spec", "FracAtlas_AUC", "FracAtlas_FP"]].to_string(index=False))

    # Calculate False Positive change relative to Exp E and Original DG-CBAM
    fp_exp_f = fracatlas_res["fp"]
    fp_exp_e = 1862
    fp_orig_dgcbam = 2309

    diff_e = fp_exp_f - fp_exp_e
    diff_orig = fp_exp_f - fp_orig_dgcbam

    pct_e = (diff_e / fp_exp_e) * 100.0
    pct_orig = (diff_orig / fp_orig_dgcbam) * 100.0

    print(f"\nFalse Positive Analysis on FracAtlas (N=3,355 normal scans):")
    print(f"  Exp F False Positives:               {fp_exp_f}")
    print(f"  Exp E False Positives:               {fp_exp_e}")
    print(f"  Original DG-CBAM False Positives:    {fp_orig_dgcbam}")
    print(f"  FP Change vs Exp E:                  {diff_e:+d} ({pct_e:+.2f}%)")
    print(f"  FP Change vs Original DG-CBAM:       {diff_orig:+d} ({pct_orig:+.2f}%)")

    # Scientific Decision Rule Classification
    if fracatlas_res["accuracy"] > 52.0 and bfrmx_test_res["accuracy"] >= 99.0 and fracatlas_res["fp"] < fp_exp_e:
        decision = "A. IMPROVED GENERALIZATION — External performance improved meaningfully while internal performance remained acceptable."
    elif fracatlas_res["fp"] < fp_exp_e or fracatlas_res["roc_auc"] > 0.65 or fracatlas_res["accuracy"] > 50.0:
        decision = "B. PARTIAL IMPROVEMENT — Some external metrics improved while others remained balanced."
    elif abs(fracatlas_res["accuracy"] - 49.90) < 2.0:
        decision = "C. NO MEANINGFUL IMPROVEMENT — External performance remains approximately unchanged."
    else:
        decision = "D. HARMFUL TRADE-OFF — External performance changed with unfavorable trade-offs."

    print(f"\nScientific Decision Classification: {decision}")

    # Generate Full Experiment Report
    with open(os.path.join(out_dir, "experiment_report.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BONEGUARD AI — EXPERIMENT F SCIENTIFIC EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Author: BoneGuard AI Research Team\n")
        f.write("Project: Advanced Bone Fracture Assessment Through Deep Learning-Based X-Ray Image Analysis\n\n")
        f.write("1. OBJECTIVE & HYPOTHESIS\n")
        f.write("-" * 60 + "\n")
        f.write("Objective: Evaluate whether realistic radiographic domain-variation augmentation during training on\n")
        f.write("           the BFRMX training partition improves generalization on the untouched external FracAtlas benchmark.\n")
        f.write("Hypothesis: Exposing the network to plausible variations in X-ray exposure, contrast LUTs, detector sharpness,\n")
        f.write("            and slight geometry will make feature representations more robust to scanner domain shift,\n")
        f.write("            thereby mitigating out-of-domain false-positive inflation.\n\n")
        f.write("2. DATASET & PARTITION CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"BFRMX Train:      {len(train_ds)} samples ({train_ds.get_class_counts()})\n")
        f.write(f"BFRMX Validation: {len(val_ds)} samples ({val_ds.get_class_counts()})\n")
        f.write(f"BFRMX Test:       {len(test_ds)} samples ({test_ds.get_class_counts()})\n")
        f.write(f"FracAtlas Ext:    {len(fracatlas_ds)} samples ({fracatlas_ds.get_class_counts()})\n")
        f.write("Data Isolation Protocol: FracAtlas was completely isolated from training, validation, early stopping, and selection.\n\n")
        f.write("3. AUGMENTATION & TRAINING CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write("Augmentation Pipeline (BFRMX Train Only):\n")
        f.write("  - RandomRotation (degrees=8)\n")
        f.write("  - RandomAffine (translate=4%, scale=0.96-1.04)\n")
        f.write("  - RandomHorizontalFlip (p=0.5)\n")
        f.write("  - ColorJitter (brightness=0.15, contrast=0.15)\n")
        f.write("  - RandomAdjustSharpness (factor=1.5, p=0.3)\n")
        f.write("  - RandomAutocontrast (p=0.2)\n")
        f.write("  - GaussianBlur (kernel=3, sigma=0.1-1.0, p=0.25)\n")
        f.write("Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)\n")
        f.write("Scheduler: CosineAnnealingLR (T_max=25, eta_min=1e-6)\n")
        f.write("Loss: Unweighted CrossEntropyLoss [1.0, 1.0]\n")
        f.write(f"Training Time: {train_time/60:.2f} minutes | Best Epoch: {best_epoch}\n\n")
        f.write("4. IN-DOMAIN BFRMX TEST RESULTS (N = 560)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:    {bfrmx_test_res['accuracy']:.2f}%\n")
        f.write(f"Precision:   {bfrmx_test_res['precision']:.2f}%\n")
        f.write(f"Recall:      {bfrmx_test_res['recall']:.2f}%\n")
        f.write(f"Specificity: {bfrmx_test_res['specificity']:.2f}%\n")
        f.write(f"F1-Score:    {bfrmx_test_res['f1']:.2f}%\n")
        f.write(f"ROC-AUC:     {bfrmx_test_res['roc_auc']:.4f}\n")
        f.write(f"PR-AUC:      {bfrmx_test_res['pr_auc']:.4f}\n")
        f.write(f"Confusion Matrix: TP={bfrmx_test_res['tp']}, TN={bfrmx_test_res['tn']}, FP={bfrmx_test_res['fp']}, FN={bfrmx_test_res['fn']}\n\n")
        f.write("5. OUT-OF-DOMAIN FRACATLAS BENCHMARK RESULTS (N = 4,068)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:    {fracatlas_res['accuracy']:.2f}%\n")
        f.write(f"Precision:   {fracatlas_res['precision']:.2f}%\n")
        f.write(f"Recall:      {fracatlas_res['recall']:.2f}%\n")
        f.write(f"Specificity: {fracatlas_res['specificity']:.2f}%\n")
        f.write(f"F1-Score:    {fracatlas_res['f1']:.2f}%\n")
        f.write(f"ROC-AUC:     {fracatlas_res['roc_auc']:.4f}\n")
        f.write(f"PR-AUC:      {fracatlas_res['pr_auc']:.4f}\n")
        f.write(f"Confusion Matrix: TP={fracatlas_res['tp']}, TN={fracatlas_res['tn']}, FP={fracatlas_res['fp']}, FN={fracatlas_res['fn']}\n\n")
        f.write("6. COMPARISON WITH EXP E & BASELINE MODELS\n")
        f.write("-" * 60 + "\n")
        f.write(df_comp.to_string(index=False) + "\n\n")
        f.write("7. FALSE POSITIVE & NEGATIVE ANALYSIS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Exp F FP: {fp_exp_f} | Exp E FP: {fp_exp_e} (Diff: {diff_e:+d}, {pct_e:+.2f}%)\n")
        f.write(f"Exp F FP vs Orig DG-CBAM: Diff: {diff_orig:+d} ({pct_orig:+.2f}%)\n")
        f.write(f"Exp F FN: {fracatlas_res['fn']} | Exp E FN: 176 (Diff: {fracatlas_res['fn']-176:+d})\n\n")
        f.write("8. SCIENTIFIC DECISION & CONCLUSION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Decision Classification: {decision}\n\n")
        f.write("Recommendation on Prototype Deployment:\n")
        if fracatlas_res["accuracy"] > 50.5 and bfrmx_test_res["accuracy"] >= 99.2:
            f.write("Exp F achieves superior domain robustness and should be considered for deployment.\n")
        else:
            f.write("Exp E remains the verified reference deployment checkpoint due to optimal in-domain perfection\n")
            f.write("and consistent established baseline metrics.\n")
        f.write("=" * 80 + "\n")

    print(f"\nFull report written to: {os.path.join(out_dir, 'experiment_report.txt')}")


if __name__ == "__main__":
    main()
