"""
BoneGuard AI - DenseNet121 with Disagreement-Guided CBAM (DG-CBAM) Architecture
================================================================================
Model 3: Standard DenseNet121 integrated with the proposed Disagreement-Guided
Convolutional Block Attention Module (DG-CBAM).

Proposed Research Formulation:
- Channel Attention Map: Mc in R^{B x C x 1 x 1}
- Spatial Attention Map: Ms in R^{B x 1 x H x W}
- Channel-Spatial Disagreement Map: D = |Mc - Ms| in R^{B x C x H x W}
- Refined Attention Map: M_new = Ms + gamma * D in R^{B x C x H x W}
- Refined Feature Map: F_out = F (x) M_new
- Learnable Parameter: gamma (initialized to 0.1)

Placement: Inserted at the output of each DenseBlock (denseblock 1, 2, 3, 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class ChannelAttention(nn.Module):
    """
    Standard Channel Attention Module (Woo et al., 2018).
    Applies Average Pooling and Max Pooling across spatial dimensions,
    passes both through a shared two-layer MLP with reduction ratio r=16,
    and applies Sigmoid activation to produce channel attention weights Mc.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        mc = self.sigmoid(avg_out + max_out)
        return mc  # [B, C, 1, 1]


class SpatialAttention(nn.Module):
    """
    Standard Spatial Attention Module (Woo et al., 2018).
    Applies Average Pooling and Max Pooling along the channel axis,
    concatenates the two 1-channel maps into a 2-channel tensor,
    applies a 7x7 Convolution followed by Sigmoid activation to produce
    spatial attention weights Ms.
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
        ms = self.sigmoid(self.conv(concat))
        return ms  # [B, 1, H, W]


class DGCBAM(nn.Module):
    """
    Proposed Disagreement-Guided Convolutional Block Attention Module (DG-CBAM).
    Computes Channel Attention Mc and Spatial Attention Ms directly from input F,
    calculates the broadcasted Channel-Spatial Disagreement D = |Mc - Ms|,
    and refines the attention via M_new = Ms + gamma * D, where gamma is a learnable parameter.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7, initial_gamma: float = 0.1):
        super(DGCBAM, self).__init__()
        self.in_channels = in_channels
        self.ca = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        # Learnable scalar parameter gamma initialized to 0.1
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        mc = self.ca(x)  # [B, C, 1, 1]
        ms = self.sa(x)  # [B, 1, H, W]

        # Calculate Channel-Spatial Disagreement map D = |Mc - Ms|
        # Broadcasts Mc [B, C, 1, 1] and Ms [B, 1, H, W] to [B, C, H, W]
        d = torch.abs(mc - ms)  # [B, C, H, W]

        # Refine attention map M_new = Ms + gamma * D
        m_new = ms + self.gamma * d  # [B, C, H, W]

        # Feature refinement F_out = F (x) M_new
        f_out = x * m_new

        if return_maps:
            maps = {
                "Mc": mc.detach(),
                "Ms": ms.detach(),
                "D": d.detach(),
                "M_new": m_new.detach(),
                "gamma": self.gamma.item()
            }
            return f_out, maps

        return f_out


class DenseNet121_DGCBAM(nn.Module):
    """
    DenseNet-121 backbone with proposed DG-CBAM modules inserted at the output
    of each of the 4 DenseBlocks before transition/norm layers.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, initial_gamma: float = 0.1):
        super(DenseNet121_DGCBAM, self).__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = densenet121(weights=weights)
        
        # DenseNet feature extractor components
        self.features = base_model.features
        
        # Proposed DG-CBAM modules placed after each DenseBlock
        # DenseBlock 1 output channels: 256
        # DenseBlock 2 output channels: 512
        # DenseBlock 3 output channels: 1024
        # DenseBlock 4 output channels: 1024
        self.dgcbam1 = DGCBAM(in_channels=256, reduction_ratio=16, kernel_size=7, initial_gamma=initial_gamma)
        self.dgcbam2 = DGCBAM(in_channels=512, reduction_ratio=16, kernel_size=7, initial_gamma=initial_gamma)
        self.dgcbam3 = DGCBAM(in_channels=1024, reduction_ratio=16, kernel_size=7, initial_gamma=initial_gamma)
        self.dgcbam4 = DGCBAM(in_channels=1024, reduction_ratio=16, kernel_size=7, initial_gamma=initial_gamma)
        
        # 2-class Linear classification head (identical to baseline and standard CBAM)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor, return_attention_maps: bool = False):
        attention_maps = {} if return_attention_maps else None

        # Initial stem
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        # Stage 1: DenseBlock 1 -> DG-CBAM 1 -> Transition 1
        x = self.features.denseblock1(x)
        if return_attention_maps:
            x, maps1 = self.dgcbam1(x, return_maps=True)
            attention_maps["dgcbam1"] = maps1
        else:
            x = self.dgcbam1(x)
        x = self.features.transition1(x)

        # Stage 2: DenseBlock 2 -> DG-CBAM 2 -> Transition 2
        x = self.features.denseblock2(x)
        if return_attention_maps:
            x, maps2 = self.dgcbam2(x, return_maps=True)
            attention_maps["dgcbam2"] = maps2
        else:
            x = self.dgcbam2(x)
        x = self.features.transition2(x)

        # Stage 3: DenseBlock 3 -> DG-CBAM 3 -> Transition 3
        x = self.features.denseblock3(x)
        if return_attention_maps:
            x, maps3 = self.dgcbam3(x, return_maps=True)
            attention_maps["dgcbam3"] = maps3
        else:
            x = self.dgcbam3(x)
        x = self.features.transition3(x)

        # Stage 4: DenseBlock 4 -> DG-CBAM 4 -> Final Norm & ReLU
        x = self.features.denseblock4(x)
        if return_attention_maps:
            x, maps4 = self.dgcbam4(x, return_maps=True)
            attention_maps["dgcbam4"] = maps4
        else:
            x = self.dgcbam4(x)
        x = self.features.norm5(x)
        x = F.relu(x, inplace=True)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Classification Logits
        out = self.classifier(x)

        if return_attention_maps:
            return out, attention_maps
        return out

    def get_gamma_values(self) -> dict:
        """
        Returns the current learned gamma scalar values for each DG-CBAM module.
        """
        return {
            "gamma_stage1": float(self.dgcbam1.gamma.item()),
            "gamma_stage2": float(self.dgcbam2.gamma.item()),
            "gamma_stage3": float(self.dgcbam3.gamma.item()),
            "gamma_stage4": float(self.dgcbam4.gamma.item()),
            "gamma_mean": float(
                (self.dgcbam1.gamma.item() + self.dgcbam2.gamma.item() + 
                 self.dgcbam3.gamma.item() + self.dgcbam4.gamma.item()) / 4.0
            )
        }


def build_densenet121_dgcbam(num_classes: int = 2, pretrained: bool = True, initial_gamma: float = 0.1) -> nn.Module:
    """
    Factory function to construct DenseNet121 + DG-CBAM.
    """
    return DenseNet121_DGCBAM(num_classes=num_classes, pretrained=pretrained, initial_gamma=initial_gamma)


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
    model = build_densenet121_dgcbam(num_classes=2, pretrained=True, initial_gamma=0.1)
    counts = get_model_parameter_counts(model)
    print("DenseNet121 + DG-CBAM instantiated successfully.")
    print(f"Total parameters:     {counts['total_params']:,}")
    print(f"Trainable parameters: {counts['trainable_params']:,}")
    print(f"Initial Gammas:       {model.get_gamma_values()}")
    
    # Test dummy input pass with shapes verification
    dummy_input = torch.randn(4, 3, 224, 224)
    out, maps = model(dummy_input, return_attention_maps=True)
    print(f"Dummy forward pass output shape: {out.shape} (Expected: [4, 2])")
    for k, v in maps.items():
        print(f"{k} shapes -> Mc: {list(v['Mc'].shape)}, Ms: {list(v['Ms'].shape)}, D: {list(v['D'].shape)}, M_new: {list(v['M_new'].shape)}")
