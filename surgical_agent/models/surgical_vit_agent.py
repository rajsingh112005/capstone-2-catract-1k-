
import torch
import torch.nn as nn
import timm


class SurgicalViTAgent(nn.Module):
    """Vision Transformer-based model for surgical phase recognition with temporal modeling."""

    def __init__(self, num_classes):
        """
        Initialize the Surgical ViT Agent.

        Args:
            num_classes (int): Number of surgical phases to classify.
        """
        super(SurgicalViTAgent, self).__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        self.temporal_head = nn.GRU(768, 256, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, frames, channels, height, width).

        Returns:
            torch.Tensor: Classification logits of shape (batch, num_classes).
        """
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        features = self.backbone(x)
        features = features.view(b, f, -1)
        _, h_n = self.temporal_head(features)
        return self.classifier(h_n[-1])
