"""
BoneGuard AI - DenseNet121 Baseline Comprehensive Evaluation Pipeline
======================================================================
Step 1: Evaluate the frozen best DenseNet121 checkpoint on:
1. BFRMX Internal Test Set (Unseen internal test partition)
2. FracAtlas External Benchmark Set (100% external unseen hospital data)

Calculates: Accuracy, Precision, Recall, F1-score, ROC-AUC, TP, TN, FP, FN,
and generates publication-ready confusion matrices and ROC curves.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# Add parent directory to path to import model module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.densenet121_baseline import build_densenet121_baseline
from training.train_densenet121 import BoneGuardXRayDataset, get_data_transforms


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion Matrix"):
    """
    Plots and saves a professional confusion matrix heatmap.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Label cells with raw count and percentage
    total = np.sum(cm)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / total) * 100
            plt.text(
                j, i, f"{count:,}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    plt.ylabel('True Class Ground Truth', fontsize=12, fontweight='bold')
    plt.xlabel('Model Predicted Class', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, output_path, title="ROC Curve"):
    """
    Plots and saves a publication-ready ROC curve.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'DenseNet121 Baseline (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, dataset_name: str, output_dir: str):
    """
    Runs inference and calculates detailed classification metrics for a dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []

    for images, labels, fnames in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_filenames.extend(fnames)

    all_probs = np.vstack(all_probs)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    # Class 0: FRACTURE (Positive Class), Class 1: NO_FRACTURE (Negative Class)
    # y_true_bin: 1 for FRACTURE, 0 for NO_FRACTURE
    y_true_bin = (y_true == 0).astype(int)
    y_pred_bin = (y_pred == 0).astype(int)
    fracture_prob = all_probs[:, 0]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Binary metrics specifically for FRACTURE class
    prec_fracture = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec_fracture = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1_fracture = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    roc_auc = roc_auc_score(y_true_bin, fracture_prob)
    fpr, tpr, _ = roc_curve(y_true_bin, fracture_prob)

    # Confusion matrix with labels [0, 1] -> [FRACTURE, NO_FRACTURE]
    # cm[0,0] = TP (Fracture pred Fracture)
    # cm[0,1] = FN (Fracture pred No Fracture)
    # cm[1,0] = FP (No Fracture pred Fracture)
    # cm[1,1] = TN (No Fracture pred No Fracture)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "dataset": dataset_name,
        "total_samples": len(y_true),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_fracture": float(prec_fracture),
        "recall_fracture (sensitivity)": float(rec_fracture),
        "f1_fracture": float(f1_fracture),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "true_positives (TP)": tp,
        "true_negatives (TN)": tn,
        "false_positives (FP)": fp,
        "false_negatives (FN)": fn
    }

    # Plot Confusion Matrix
    prefix = dataset_name.replace(" ", "_")
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    plot_confusion_matrix(cm, ["FRACTURE", "NO_FRACTURE"], cm_path, title=f"{dataset_name} - Confusion Matrix")

    # Plot ROC Curve
    roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
    plot_roc_curve(fpr, tpr, roc_auc, roc_path, title=f"{dataset_name} - ROC Curve")

    # Save metrics CSV
    csv_path = os.path.join(output_dir, f"{prefix}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    return metrics, cm


def run_full_evaluation(
    checkpoint_path: str,
    data_split_dir: str,
    fracatlas_dir: str,
    output_dir: str,
    batch_size: int = 32
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("BONEGUARD AI: DENSENET121 BASELINE EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Loading Checkpoint: {checkpoint_path}")
    print(f"Inference Device:   {device}")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_densenet121_baseline(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_epoch = checkpoint.get("epoch", "N/A")
    print(f"Model successfully loaded (Trained at Epoch {best_epoch}).")

    # 2. Setup DataLoaders (Deterministic Evaluation Transforms Only)
    _, eval_tf = get_data_transforms()
    test_dir = os.path.join(data_split_dir, "test")
    test_ds = BoneGuardXRayDataset(test_dir, transform=eval_tf)
    frac_ds = BoneGuardXRayDataset(fracatlas_dir, transform=eval_tf)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    frac_loader = DataLoader(frac_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Evaluate Internal Test
    print("\nEvaluating on BFRMX Internal Test Set...", flush=True)
    bfrmx_metrics, bfrmx_cm = evaluate_dataset(
        model, test_loader, device, "BFRMX_internal_test", output_dir
    )

    # 4. Evaluate FracAtlas External Benchmark
    print("Evaluating on FracAtlas External Benchmark Set...")
    frac_metrics, frac_cm = evaluate_dataset(
        model, frac_loader, device, "FracAtlas_external_test", output_dir
    )

    # 5. Generate Final Comprehensive Baseline Report
    report_text = f"""================================================================================
BONEGUARD AI: DENSENET121 BASELINE FINAL EXPERIMENT REPORT
================================================================================
Project Title: "Advanced Bone Fracture Assessment and Clinical Decision Support
                Through Deep Learning-Based X-Ray Image Analysis"
Model:          Standard DenseNet121 Baseline (torchvision)
Pretrained:     ImageNet-1K Weights
Modifications:  2-Class Linear Classifier Replacement (No Attention / No CBAM)
Target Task:    Binary X-Ray Fracture Classification (FRACTURE vs NO_FRACTURE)
Evaluation Date: September 2026

==================================================
1. BFRMX INTERNAL TEST SET PERFORMANCE
==================================================
Total Test Samples:             {bfrmx_metrics['total_samples']}
Classification Accuracy:        {bfrmx_metrics['accuracy']:.4f} ({bfrmx_metrics['accuracy']*100:.2f}%)
Macro Precision:                {bfrmx_metrics['precision_macro']:.4f}
Macro Recall:                   {bfrmx_metrics['recall_macro']:.4f}
Macro F1-Score:                 {bfrmx_metrics['f1_macro']:.4f}
Fracture Precision:             {bfrmx_metrics['precision_fracture']:.4f}
Fracture Sensitivity (Recall):  {bfrmx_metrics['recall_fracture (sensitivity)']:.4f}
Fracture F1-Score:              {bfrmx_metrics['f1_fracture']:.4f}
Specificity (No-Fracture Rec):  {bfrmx_metrics['specificity']:.4f}
Area Under ROC Curve (ROC-AUC): {bfrmx_metrics['roc_auc']:.4f}

Confusion Matrix Breakdown:
- True Positives  (TP - Fracture correctly predicted):    {bfrmx_metrics['true_positives (TP)']}
- True Negatives  (TN - No-Fracture correctly predicted): {bfrmx_metrics['true_negatives (TN)']}
- False Positives (FP - No-Fracture predicted Fracture):  {bfrmx_metrics['false_positives (FP)']}
- False Negatives (FN - Fracture missed):                 {bfrmx_metrics['false_negatives (FN)']}

==================================================
2. FRACATLAS EXTERNAL BENCHMARK PERFORMANCE
==================================================
Total External Samples:         {frac_metrics['total_samples']}
Classification Accuracy:        {frac_metrics['accuracy']:.4f} ({frac_metrics['accuracy']*100:.2f}%)
Macro Precision:                {frac_metrics['precision_macro']:.4f}
Macro Recall:                   {frac_metrics['recall_macro']:.4f}
Macro F1-Score:                 {frac_metrics['f1_macro']:.4f}
Fracture Precision:             {frac_metrics['precision_fracture']:.4f}
Fracture Sensitivity (Recall):  {frac_metrics['recall_fracture (sensitivity)']:.4f}
Fracture F1-Score:              {frac_metrics['f1_fracture']:.4f}
Specificity (No-Fracture Rec):  {frac_metrics['specificity']:.4f}
Area Under ROC Curve (ROC-AUC): {frac_metrics['roc_auc']:.4f}

Confusion Matrix Breakdown:
- True Positives  (TP - Fracture correctly predicted):    {frac_metrics['true_positives (TP)']}
- True Negatives  (TN - No-Fracture correctly predicted): {frac_metrics['true_negatives (TN)']}
- False Positives (FP - No-Fracture predicted Fracture):  {frac_metrics['false_positives (FP)']}
- False Negatives (FN - Fracture missed):                 {frac_metrics['false_negatives (FN)']}

==================================================
3. CLINICAL AI & GENERALIZATION OBSERVATIONS
==================================================
- Role: This standard DenseNet121 model serves as the foundational baseline for BoneGuard AI.
- Internal Test Evaluation: Demonstrates feature extraction capability on identically acquired radiographs.
- External Generalization: FracAtlas serves as a rigorous cross-domain external validation benchmark.
- Next Steps: DenseNet121 + CBAM and DenseNet121 + DG-CBAM will be compared directly against this baseline.
================================================================================
"""
    report_path = os.path.join(output_dir, "final_baseline_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"All evaluation artifacts and visual plots saved to {output_dir}")
    return bfrmx_metrics, frac_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DenseNet121 Baseline on Test Sets")
    parser.add_argument("--checkpoint", type=str, default=r"D:\BoneGuard AI\06_Model_Results\DenseNet121_Baseline\best_model.pth")
    parser.add_argument("--split_dir", type=str, default=r"D:\BoneGuard AI\04_Preprocessing\02_Split")
    parser.add_argument("--fracatlas_dir", type=str, default=r"D:\BoneGuard AI\04_Preprocessing\01_Cleaned\FracAtlas")
    parser.add_argument("--output_dir", type=str, default=r"D:\BoneGuard AI\06_Model_Results\DenseNet121_Baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        data_split_dir=args.split_dir,
        fracatlas_dir=args.fracatlas_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
