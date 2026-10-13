"""
BoneGuard AI - DenseNet121 with Standard CBAM Architecture
===========================================================
Model 2: Standard DenseNet121 integrated with canonical Convolutional Block
Attention Module (CBAM: Channel Attention + Spatial Attention).

Adheres strictly to the standard CBAM formulation (Woo et al., ECCV 2018):
- Channel Attention: Global Avg Pool + Global Max Pool -> Shared MLP -> Sigmoid -> Mc * F
- Spatial Attention: Channel Avg Pool + Channel Max Pool -> Concat(2ch) -> 7x7 Conv -> Sigmoid -> Ms * F'
- Placement: Inserted at the output of each DenseBlock (after denseblock 1, 2, 3, 4)
- NO DG-CBAM, no disagreement maps, no custom fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class ChannelAttention(nn.Module):
    """
    Standard Channel Attention Module (Woo et al., 2018).
    Applies both Average Pooling and Max Pooling across spatial dimensions,
    passes both through a shared two-layer MLP with reduction ratio r=16,
    and applies Sigmoid activation to produce channel attention weights.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP implemented with 1x1 convolutions for seamless 4D tensor processing
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att


class SpatialAttention(nn.Module):
    """
    Standard Spatial Attention Module (Woo et al., 2018).
    Applies Average Pooling and Max Pooling along the channel axis,
    concatenates the two 1-channel maps into a 2-channel tensor,
    applies a 7x7 Convolution followed by Sigmoid activation to produce
    spatial attention weights.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(concat))
        return x * spatial_att


class CBAM(nn.Module):
    """
    Standard Convolutional Block Attention Module (Woo et al., ECCV 2018).
    Sequentially applies Channel Attention followed by Spatial Attention:
        F'  = Mc(F) * F
        F'' = Ms(F') * F'
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DenseNet121_CBAM(nn.Module):
    """
    DenseNet-121 backbone with standard CBAM modules inserted at the output
    of each of the 4 DenseBlocks before transition/norm layers.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(DenseNet121_CBAM, self).__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = densenet121(weights=weights)
        
        # DenseNet feature components
        self.features = base_model.features
        
        # Standard CBAM modules placed after each DenseBlock
        # DenseBlock 1 output channels: 256
        # DenseBlock 2 output channels: 512
        # DenseBlock 3 output channels: 1024
        # DenseBlock 4 output channels: 1024
        self.cbam1 = CBAM(in_channels=256, reduction_ratio=16, kernel_size=7)
        self.cbam2 = CBAM(in_channels=512, reduction_ratio=16, kernel_size=7)
        self.cbam3 = CBAM(in_channels=1024, reduction_ratio=16, kernel_size=7)
        self.cbam4 = CBAM(in_channels=1024, reduction_ratio=16, kernel_size=7)
        
        # 2-class Linear classification head (identical to baseline)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial stem
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        # Stage 1: DenseBlock 1 -> CBAM 1 -> Transition 1
        x = self.features.denseblock1(x)
        x = self.cbam1(x)
        x = self.features.transition1(x)

        # Stage 2: DenseBlock 2 -> CBAM 2 -> Transition 2
        x = self.features.denseblock2(x)
        x = self.cbam2(x)
        x = self.features.transition2(x)

        # Stage 3: DenseBlock 3 -> CBAM 3 -> Transition 3
        x = self.features.denseblock3(x)
        x = self.cbam3(x)
        x = self.features.transition3(x)

        # Stage 4: DenseBlock 4 -> CBAM 4 -> Final Norm & ReLU
        x = self.features.denseblock4(x)
        x = self.cbam4(x)
        x = self.features.norm5(x)
        x = F.relu(x, inplace=True)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Classification Logits
        x = self.classifier(x)
        return x


def build_densenet121_cbam(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function to construct DenseNet121 + Standard CBAM.
    """
    return DenseNet121_CBAM(num_classes=num_classes, pretrained=pretrained)


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
    model = build_densenet121_cbam(num_classes=2, pretrained=True)
    counts = get_model_parameter_counts(model)
    print("DenseNet121 + Standard CBAM instantiated successfully.")
    print(f"Total parameters:     {counts['total_params']:,}")
    print(f"Trainable parameters: {counts['trainable_params']:,}")
    
    # Test dummy input pass
    dummy_input = torch.randn(4, 3, 224, 224)
    out = model(dummy_input)
    print(f"Dummy forward pass output shape: {out.shape} (Expected: [4, 2])")
