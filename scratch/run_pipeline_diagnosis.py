"""
BoneGuard AI - Comprehensive Inference Pipeline Diagnostic Script
==================================================================
Performs step-by-step verification of:
1. Preprocessing parameters & tensor statistics
2. Class mapping consistency
3. Checkpoint verification
4. Evaluation pipeline vs FastAPI UI inference pipeline comparison on:
   - BFRMX test images
   - FracAtlas external benchmark images
5. High-confidence prediction analysis on out-of-domain scans
"""

import os
import io
import sys
import numpy as np
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "05_Code") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "05_Code"))

from models.densenet121_dgcbam import build_densenet121_dgcbam, get_model_parameter_counts
from training.train_densenet121 import BoneGuardXRayDataset
from backend.inference.pipeline import get_inference_service, BoneGuardInferenceService

def main():
    print("=" * 80)
    print("BONEGUARD AI — INFERENCE PIPELINE & DOMAIN SHIFT DIAGNOSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")

    # 1. Checkpoint Verification
    ckpt_path = os.path.join(PROJECT_ROOT, "06_Model_Results", "DGCBAM_DomainRobustness", "Exp_E_ClassWeight", "best_model.pth")
    print(f"\n1. CHECKPOINT AUDIT")
    print(f"Path: {ckpt_path}")
    print(f"Exists: {os.path.exists(ckpt_path)}")
    ckpt_data = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt_data, dict):
        print(f"Checkpoint keys: {list(ckpt_data.keys())}")
        print(f"Best Epoch: {ckpt_data.get('best_epoch') or ckpt_data.get('epoch')}")
        print(f"Experiment: {ckpt_data.get('experiment_name')}")
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    else:
        state_dict = ckpt_data

    # 2. Evaluation Pipeline Model Setup
    eval_model = build_densenet121_dgcbam(num_classes=2, pretrained=False, initial_gamma=0.1)
    eval_model.load_state_dict(state_dict)
    eval_model.to(device)
    eval_model.eval()

    # 3. UI Backend Inference Service Setup
    ui_service = get_inference_service()

    print("\n2. PARAMETER & WEIGHT COMPARISON (Eval Model vs UI Model)")
    eval_params = [p.data for p in eval_model.parameters()]
    ui_params = [p.data for p in ui_service.model.parameters()]
    param_match = all(torch.equal(p1, p2) for p1, p2 in zip(eval_params, ui_params))
    print(f"Exact Parameter Match: {param_match}")

    # 4. Preprocessing Transforms Audit
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ui_transform = ui_service.transform
    print("\n3. PREPROCESSING PROTOCOL AUDIT")
    print(f"Eval Transform: {eval_transform}")
    print(f"UI Transform:   {ui_transform}")

    # 5. Class Mapping Audit
    print("\n4. CLASS MAPPING AUDIT")
    bfrmx_test_dir = os.path.join(PROJECT_ROOT, "04_Preprocessing", "02_Split", "test")
    fracatlas_dir = os.path.join(PROJECT_ROOT, "04_Preprocessing", "01_Cleaned", "FracAtlas")

    eval_ds = BoneGuardXRayDataset(bfrmx_test_dir, transform=eval_transform)
    print(f"Training/Eval Dataset Class-to-Idx: {eval_ds.class_to_idx}")
    print(f"Training/Eval Dataset Idx-to-Class: {eval_ds.idx_to_class}")
    print(f"UI Service Class Names:             {ui_service.class_names}")
    class_mapping_match = (eval_ds.class_to_idx["FRACTURE"] == 0 and ui_service.class_names[0] == "FRACTURE" and
                           eval_ds.class_to_idx["NO_FRACTURE"] == 1 and ui_service.class_names[1] == "NO_FRACTURE")
    print(f"Class Mapping Exact Match:          {class_mapping_match}")

    # 6. Sample-by-Sample Comparison on BFRMX Test Images
    print("\n5. BFRMX TEST SAMPLES: EVALUATION PIPELINE VS UI PIPELINE")
    bfrmx_frac_samples = [os.path.join(bfrmx_test_dir, "FRACTURE", f) for f in os.listdir(os.path.join(bfrmx_test_dir, "FRACTURE"))[:10]]
    bfrmx_norm_samples = [os.path.join(bfrmx_test_dir, "NO_FRACTURE", f) for f in os.listdir(os.path.join(bfrmx_test_dir, "NO_FRACTURE"))[:10]]

    max_tensor_diff = 0.0
    max_logit_diff = 0.0
    max_prob_diff = 0.0

    print("\n--- BFRMX Fracture Samples ---")
    for path in bfrmx_frac_samples[:5]:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Method A: Eval pipeline
        img_a = Image.open(path).convert("RGB")
        tensor_a = eval_transform(img_a).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_a = eval_model(tensor_a).cpu().numpy()[0]
            probs_a = torch.softmax(torch.tensor(logits_a), dim=0).numpy()
            pred_a = "FRACTURE" if probs_a[0] >= 0.50 else "NO FRACTURE"

        # Method B: UI pipeline
        tensor_b, _, _ = ui_service.preprocess_image(raw_bytes)
        ui_res = ui_service.predict(raw_bytes)
        probs_b = np.array([ui_res["fracture_probability"], ui_res["no_fracture_probability"]])
        pred_b = ui_res["prediction"]

        t_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
        l_diff = np.max(np.abs(logits_a - ui_service.model(tensor_b).cpu().detach().numpy()[0]))
        p_diff = np.max(np.abs(probs_a - probs_b))

        max_tensor_diff = max(max_tensor_diff, t_diff)
        max_logit_diff = max(max_logit_diff, l_diff)
        max_prob_diff = max(max_prob_diff, p_diff)

        print(f"File: {os.path.basename(path)}")
        print(f"  Eval -> Pred: {pred_a:<11} | P(Frac): {probs_a[0]:.4f} | P(Norm): {probs_a[1]:.4f} | Logits: [{logits_a[0]:.4f}, {logits_a[1]:.4f}]")
        print(f"  UI   -> Pred: {pred_b:<11} | P(Frac): {probs_b[0]:.4f} | P(Norm): {probs_b[1]:.4f} | Tensor Diff: {t_diff:.2e}")
        assert pred_a == pred_b, f"Prediction mismatch on {path}"

    print("\n--- BFRMX Normal Samples ---")
    for path in bfrmx_norm_samples[:5]:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        img_a = Image.open(path).convert("RGB")
        tensor_a = eval_transform(img_a).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_a = eval_model(tensor_a).cpu().numpy()[0]
            probs_a = torch.softmax(torch.tensor(logits_a), dim=0).numpy()
            pred_a = "FRACTURE" if probs_a[0] >= 0.50 else "NO FRACTURE"

        tensor_b, _, _ = ui_service.preprocess_image(raw_bytes)
        ui_res = ui_service.predict(raw_bytes)
        probs_b = np.array([ui_res["fracture_probability"], ui_res["no_fracture_probability"]])
        pred_b = ui_res["prediction"]

        t_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
        l_diff = np.max(np.abs(logits_a - ui_service.model(tensor_b).cpu().detach().numpy()[0]))
        p_diff = np.max(np.abs(probs_a - probs_b))

        max_tensor_diff = max(max_tensor_diff, t_diff)
        max_logit_diff = max(max_logit_diff, l_diff)
        max_prob_diff = max(max_prob_diff, p_diff)

        print(f"File: {os.path.basename(path)}")
        print(f"  Eval -> Pred: {pred_a:<11} | P(Frac): {probs_a[0]:.4f} | P(Norm): {probs_a[1]:.4f} | Logits: [{logits_a[0]:.4f}, {logits_a[1]:.4f}]")
        print(f"  UI   -> Pred: {pred_b:<11} | P(Frac): {probs_b[0]:.4f} | P(Norm): {probs_b[1]:.4f} | Tensor Diff: {t_diff:.2e}")
        assert pred_a == pred_b, f"Prediction mismatch on {path}"

    # 7. Sample-by-Sample Comparison on External FracAtlas Images
    print("\n6. FRACATLAS EXTERNAL BENCHMARK: EVALUATION PIPELINE VS UI PIPELINE")
    fracatlas_frac_samples = [os.path.join(fracatlas_dir, "FRACTURE", f) for f in os.listdir(os.path.join(fracatlas_dir, "FRACTURE"))[:10]]
    fracatlas_norm_samples = [os.path.join(fracatlas_dir, "NO_FRACTURE", f) for f in os.listdir(os.path.join(fracatlas_dir, "NO_FRACTURE"))[:10]]

    print("\n--- FracAtlas External Fracture Samples ---")
    for path in fracatlas_frac_samples[:5]:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        img_a = Image.open(path).convert("RGB")
        tensor_a = eval_transform(img_a).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_a = eval_model(tensor_a).cpu().numpy()[0]
            probs_a = torch.softmax(torch.tensor(logits_a), dim=0).numpy()
            pred_a = "FRACTURE" if probs_a[0] >= 0.50 else "NO FRACTURE"

        tensor_b, _, _ = ui_service.preprocess_image(raw_bytes)
        ui_res = ui_service.predict(raw_bytes)
        probs_b = np.array([ui_res["fracture_probability"], ui_res["no_fracture_probability"]])
        pred_b = ui_res["prediction"]

        t_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
        l_diff = np.max(np.abs(logits_a - ui_service.model(tensor_b).cpu().detach().numpy()[0]))
        p_diff = np.max(np.abs(probs_a - probs_b))

        max_tensor_diff = max(max_tensor_diff, t_diff)
        max_logit_diff = max(max_logit_diff, l_diff)
        max_prob_diff = max(max_prob_diff, p_diff)

        print(f"File: {os.path.basename(path)}")
        print(f"  Eval -> Pred: {pred_a:<11} | P(Frac): {probs_a[0]:.4f} | P(Norm): {probs_a[1]:.4f} | Logits: [{logits_a[0]:.4f}, {logits_a[1]:.4f}]")
        print(f"  UI   -> Pred: {pred_b:<11} | P(Frac): {probs_b[0]:.4f} | P(Norm): {probs_b[1]:.4f} | Tensor Diff: {t_diff:.2e}")
        assert pred_a == pred_b, f"Prediction mismatch on {path}"

    print("\n--- FracAtlas External Normal Samples ---")
    for path in fracatlas_norm_samples[:5]:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        img_a = Image.open(path).convert("RGB")
        tensor_a = eval_transform(img_a).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_a = eval_model(tensor_a).cpu().numpy()[0]
            probs_a = torch.softmax(torch.tensor(logits_a), dim=0).numpy()
            pred_a = "FRACTURE" if probs_a[0] >= 0.50 else "NO FRACTURE"

        tensor_b, _, _ = ui_service.preprocess_image(raw_bytes)
        ui_res = ui_service.predict(raw_bytes)
        probs_b = np.array([ui_res["fracture_probability"], ui_res["no_fracture_probability"]])
        pred_b = ui_res["prediction"]

        t_diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
        l_diff = np.max(np.abs(logits_a - ui_service.model(tensor_b).cpu().detach().numpy()[0]))
        p_diff = np.max(np.abs(probs_a - probs_b))

        max_tensor_diff = max(max_tensor_diff, t_diff)
        max_logit_diff = max(max_logit_diff, l_diff)
        max_prob_diff = max(max_prob_diff, p_diff)

        print(f"File: {os.path.basename(path)}")
        print(f"  Eval -> Pred: {pred_a:<11} | P(Frac): {probs_a[0]:.4f} | P(Norm): {probs_a[1]:.4f} | Logits: [{logits_a[0]:.4f}, {logits_a[1]:.4f}]")
        print(f"  UI   -> Pred: {pred_b:<11} | P(Frac): {probs_b[0]:.4f} | P(Norm): {probs_b[1]:.4f} | Tensor Diff: {t_diff:.2e}")
        assert pred_a == pred_b, f"Prediction mismatch on {path}"

    print("\n7. OVERALL NUMERICAL CONVERGENCE")
    print(f"Maximum Input Tensor Difference: {max_tensor_diff:.2e}")
    print(f"Maximum Output Logit Difference: {max_logit_diff:.2e}")
    print(f"Maximum Probability Difference:  {max_prob_diff:.2e}")

    # 8. Detailed Analysis of External Misclassifications
    print("\n8. EXTERNAL MISCLASSIFICATION & CONFIDENCE PROFILE ANALYSIS")
    print("Investigating why external scans are misclassified with high confidence...")

    fracatlas_all_norm = [os.path.join(fracatlas_dir, "NO_FRACTURE", f) for f in os.listdir(os.path.join(fracatlas_dir, "NO_FRACTURE"))]
    fp_high_conf = []
    fp_total = 0
    tested_count = min(300, len(fracatlas_all_norm))

    for path in fracatlas_all_norm[:tested_count]:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        res = ui_service.predict(raw_bytes)
        if res["prediction"] == "FRACTURE":
            fp_total += 1
            if res["fracture_probability"] > 0.90:
                fp_high_conf.append((os.path.basename(path), res["fracture_probability"]))

    print(f"Tested {tested_count} FracAtlas NO_FRACTURE scans:")
    print(f"  Total False Positives (predicted FRACTURE): {fp_total} ({fp_total/tested_count*100:.1f}%)")
    print(f"  High-confidence False Positives (P > 0.90): {len(fp_high_conf)} ({len(fp_high_conf)/tested_count*100:.1f}%)")
    print(f"  Top 5 High-Confidence False Positive examples:")
    for fname, prob in fp_high_conf[:5]:
        print(f"    - {fname}: P(FRACTURE) = {prob*100:.2f}%")

if __name__ == "__main__":
    main()
