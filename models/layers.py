# layers.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A Residual Block that adds skip connections within the encoder and decoder.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class AttentionBlock(nn.Module):
    """
    An Attention Block that learns spatial attention weights to focus on important regions of the feature maps.
    """
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)
