# layers.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    An optimized Residual Block using depthwise separable convolutions 
    to reduce the number of parameters and memory usage.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.block = nn.Sequential(
            self.depthwise_conv,
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            self.pointwise_conv,
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class AttentionBlock(nn.Module):
    """
    An optimized Attention Block that learns spatial attention weights 
    using grouped convolutions to reduce memory usage and computational cost.
    """
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 16, kernel_size=1, groups=4),  # Reduce intermediate channels and add groups
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1, groups=4),  # Use grouped convs for reduced memory
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)
