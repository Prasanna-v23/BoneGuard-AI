"""
BoneGuard AI - DenseNet121 Baseline Architecture
================================================
Standard DenseNet121 for Binary X-Ray Bone Fracture Classification (FRACTURE / NO_FRACTURE).
No attention mechanisms (CBAM / DG-CBAM) or custom modifications are added to this baseline.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def build_densenet121_baseline(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Constructs a standard DenseNet121 model with ImageNet pre-trained weights
    and a modified 2-class linear classification head.

    Args:
        num_classes: Number of target classes (2: FRACTURE, NO_FRACTURE).
        pretrained: If True, loads official ImageNet-1K pretrained weights.

    Returns:
        nn.Module: Standard DenseNet121 baseline model.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    # In DenseNet121, the classifier is a Linear layer taking 1024 features to num_classes
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def get_model_parameter_counts(model: nn.Module) -> dict:
    """
    Calculates total, trainable, and non-trainable parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }


if __name__ == "__main__":
    # Self-test instantiation
    model = build_densenet121_baseline(num_classes=2, pretrained=True)
    counts = get_model_parameter_counts(model)
    print("DenseNet121 Baseline instantiated successfully.")
    print(f"Total parameters: {counts['total_params']:,}")
    print(f"Trainable parameters: {counts['trainable_params']:,}")
    
    # Test dummy input pass
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"Dummy forward pass output shape: {out.shape} (Expected: [2, 2])")
