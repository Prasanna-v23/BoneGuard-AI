"""
BoneGuard AI - Backend Inference & Grad-CAM Service Pipeline
============================================================
Reuses the verified DenseNet121 + DG-CBAM model architecture and checkpoint
without modifying weights, architectures, or preprocessing protocols.
"""

import os
import sys
import io
import base64
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Ensure access to 05_Code modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "05_Code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.densenet121_dgcbam import build_densenet121_dgcbam, get_model_parameter_counts

logger = logging.getLogger("boneguard.inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class GradCAMEngine:
    """
    Grad-CAM engine hooking the verified target layer (model.features.norm5)
    for model interpretability and attention map extraction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            if self.activations.requires_grad:
                self.activations.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        self.hook_handles.append(h1)

    def generate(self, input_tensor: torch.Tensor, target_class: int = 0) -> Tuple[np.ndarray, np.ndarray, int, float, list]:
        """
        Executes Grad-CAM backpropagation on target_class.
        Returns: (heatmap_normalized, probs, pred_class, confidence, raw_probs)
        """
        self.model.eval()
        self.model.zero_grad()

        with torch.enable_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
            pred_class = int(torch.argmax(output, dim=1).cpu().item())
            confidence = float(probs[pred_class])

            # Backprop for target class
            score = output[0, target_class]
            score.backward(retain_graph=False)

        # Gradients from activation
        if self.activations is not None and self.activations.grad is not None:
            grads = self.activations.grad
        elif self.gradients is not None:
            grads = self.gradients
        else:
            grads = torch.ones_like(self.activations)

        # Global average pooling over spatial dimensions
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        heatmap_resized = cv2.resize(cam_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        return heatmap_resized, probs, pred_class, confidence, probs.tolist()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


class BoneGuardInferenceService:
    """
    Singleton service managing the DenseNet121 + DG-CBAM model lifecycle,
    single-image preprocessing, predictions, and Grad-CAM visualizations.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BoneGuardInferenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path: Optional[str] = None, device_override: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.device = torch.device(
            device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Primary deployment checkpoint: Exp_E_ClassWeight
        default_ckpt = os.path.join(
            PROJECT_ROOT, "06_Model_Results", "DGCBAM_DomainRobustness", "Exp_E_ClassWeight", "best_model.pth"
        )
        self.checkpoint_path = checkpoint_path or default_ckpt

        self.class_names = {0: "FRACTURE", 1: "NO_FRACTURE"}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = None
        self.gradcam_engine = None
        self.model_info = {}
        self._load_model()
        self._initialized = True

    def _load_model(self):
        logger.info("Initializing BoneGuard AI Inference Engine...")
        logger.info(f"Target Device: {self.device}")
        logger.info(f"Loading Checkpoint: {self.checkpoint_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        # Instantiate verified DG-CBAM architecture
        self.model = build_densenet121_dgcbam(num_classes=2, pretrained=False, initial_gamma=0.1)

        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
            best_epoch = checkpoint_data.get("epoch") or checkpoint_data.get("best_epoch", "N/A")
            exp_name = checkpoint_data.get("experiment_name", "Exp_E_ClassWeight (Unweighted CrossEntropy)")
        else:
            state_dict = checkpoint_data
            best_epoch = "N/A"
            exp_name = "DG-CBAM Model"

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        param_counts = get_model_parameter_counts(self.model)
        gamma_values = self.model.get_gamma_values()

        self.model_info = {
            "model_name": "DenseNet121 + DG-CBAM",
            "backbone": "DenseNet121 (Dense Connectivity, 4 DenseBlocks)",
            "attention_mechanism": "Difference-Guided CBAM (DG-CBAM: M_new = Ms + γ·|Mc - Ms|)",
            "num_classes": 2,
            "classes": ["FRACTURE", "NO_FRACTURE"],
            "input_size": "224x224x3",
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and self.device.type == "cuda" else "CPU",
            "total_parameters": param_counts["total_params"],
            "trainable_parameters": param_counts["trainable_params"],
            "learned_gammas": gamma_values,
            "checkpoint_path": os.path.relpath(self.checkpoint_path, PROJECT_ROOT),
            "checkpoint_size_mb": round(os.path.getsize(self.checkpoint_path) / (1024 * 1024), 2),
            "best_epoch": best_epoch,
            "experiment_name": exp_name,
            "target_gradcam_layer": "model.features.norm5 (1024 channels)"
        }

        # Initialize Grad-CAM hooked to norm5
        self.gradcam_engine = GradCAMEngine(self.model, self.model.features.norm5)

        logger.info("============================================================")
        logger.info("BoneGuard AI backend started")
        logger.info(f"Model: {self.model_info['model_name']}")
        logger.info(f"Device: {self.model_info['device']} ({self.model_info['gpu_name']})")
        logger.info(f"Checkpoint: {self.model_info['checkpoint_path']}")
        logger.info(f"Total Parameters: {self.model_info['total_parameters']:,}")
        logger.info(f"Learned Mean Gamma: {gamma_values['gamma_mean']:.4f}")
        logger.info("Model loaded successfully")
        logger.info("============================================================")

    def preprocess_image(self, image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray, Image.Image]:
        """
        Validates and converts image bytes to:
        1. PyTorch Tensor [1, 3, 224, 224] on active device
        2. RGB Numpy Array [224, 224, 3] in [0, 1] range for visualization
        3. Original PIL Image
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Unable to decode image file: {str(e)}")

        # Resized RGB image for overlay
        orig_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
        orig_rgb_np = np.array(orig_resized, dtype=np.float32) / 255.0

        # Transformed Tensor for model
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return img_tensor, orig_rgb_np, pil_img

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Executes model forward pass under torch.no_grad().
        """
        img_tensor, _, _ = self.preprocess_image(image_bytes)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        fracture_prob = float(probs[0])
        no_fracture_prob = float(probs[1])
        prediction = "FRACTURE" if fracture_prob >= 0.50 else "NO FRACTURE"
        confidence = fracture_prob if prediction == "FRACTURE" else no_fracture_prob

        return {
            "prediction": prediction,
            "predicted_class": self.class_names[pred_idx],
            "fracture_probability": round(fracture_prob, 4),
            "no_fracture_probability": round(no_fracture_prob, 4),
            "confidence": round(confidence, 4),
            "model": "DenseNet121 + DG-CBAM",
            "threshold": 0.50,
            "status": "success"
        }

    def predict_with_gradcam(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Executes model inference and generates Grad-CAM saliency visualizations.
        Returns base64 encoded images for frontend display.
        """
        img_tensor, orig_rgb_np, _ = self.preprocess_image(image_bytes)

        # Grad-CAM generated on fracture class (Class 0)
        heatmap, raw_probs, pred_idx, confidence, _ = self.gradcam_engine.generate(
            img_tensor, target_class=0
        )

        fracture_prob = float(raw_probs[0])
        no_fracture_prob = float(raw_probs[1])
        prediction = "FRACTURE" if fracture_prob >= 0.50 else "NO FRACTURE"
        pred_confidence = fracture_prob if prediction == "FRACTURE" else no_fracture_prob

        # 1. Generate Colormapped Heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # 2. Generate Overlay
        alpha = 0.45
        overlay = alpha * colored_heatmap_rgb + (1.0 - alpha) * orig_rgb_np
        overlay = np.clip(overlay, 0.0, 1.0)

        # Helper to convert RGB float np array (0..1) to Base64 PNG
        def np_to_base64_png(arr_rgb: np.ndarray) -> str:
            arr_uint8 = np.uint8(np.clip(arr_rgb * 255.0, 0, 255))
            pil = Image.fromarray(arr_uint8)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        gradcam_b64 = np_to_base64_png(overlay)
        heatmap_b64 = np_to_base64_png(colored_heatmap_rgb)
        original_b64 = np_to_base64_png(orig_rgb_np)

        return {
            "prediction": prediction,
            "predicted_class": self.class_names[pred_idx],
            "fracture_probability": round(fracture_prob, 4),
            "no_fracture_probability": round(no_fracture_prob, 4),
            "confidence": round(pred_confidence, 4),
            "model": "DenseNet121 + DG-CBAM",
            "threshold": 0.50,
            "status": "success",
            "gradcam_image": gradcam_b64,
            "heatmap_image": heatmap_b64,
            "original_image": original_b64,
            "attention_target": "Class 0 (FRACTURE)",
            "explanation": "The highlighted regions indicate areas of anatomical attention contributing to the fracture prediction. Provided for interpretability and research decision support."
        }

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info

    def get_validation_info(self) -> Dict[str, Any]:
        return {
            "internal_validation": {
                "dataset_name": "BFRMX Internal Test Partition",
                "sample_count": 560,
                "fracture_samples": 226,
                "no_fracture_samples": 334,
                "accuracy": 99.64,
                "precision": 100.00,
                "recall_sensitivity": 99.12,
                "specificity": 100.00,
                "f1_score": 99.56,
                "roc_auc": 0.9996,
                "pr_auc": 0.9995,
                "confusion_matrix": {
                    "tp": 224,
                    "tn": 334,
                    "fp": 0,
                    "fn": 2
                },
                "protocol": "Strict zero-data-leakage 70/15/15 patient split with locked 0.50 threshold."
            },
            "external_benchmark": {
                "dataset_name": "FracAtlas External Benchmark",
                "sample_count": 4068,
                "fracture_samples": 713,
                "no_fracture_samples": 3355,
                "accuracy": 49.90,
                "precision": 22.38,
                "recall_sensitivity": 75.32,
                "specificity": 44.50,
                "f1_score": 34.51,
                "roc_auc": 0.6479,
                "pr_auc": 0.2711,
                "confusion_matrix": {
                    "tp": 537,
                    "tn": 1493,
                    "fp": 1862,
                    "fn": 176
                },
                "note": "Independent external benchmark — research evaluation only. Not used for model selection. Evaluates cross-scanner generalization and domain shift."
            },
            "disclaimer": "BoneGuard AI is an investigational research prototype. Its predictions are intended for research and educational purposes and should not be used as a standalone medical diagnosis."
        }


# Global singleton helper
_service_instance: Optional[BoneGuardInferenceService] = None

def get_inference_service() -> BoneGuardInferenceService:
    global _service_instance
    if _service_instance is None:
        _service_instance = BoneGuardInferenceService()
    return _service_instance
