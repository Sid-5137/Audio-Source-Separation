import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, padding=0, transpose=False, dropout=0.1):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.padding = padding

        if self.transpose:
            self.filter = nn.ConvTranspose2d(n_inputs, n_outputs, self.kernel_size, stride, padding=self.padding)
        else:
            self.filter = nn.Conv2d(n_inputs, n_outputs, self.kernel_size, stride, padding=self.padding)

        NORM_CHANNELS = 8
        if conv_type == "gn":
            assert n_outputs % NORM_CHANNELS == 0
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm2d(n_outputs, momentum=0.01)
        else:
            self.norm = None

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        out = self.filter(x)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, negative_slope=0.2)
        if self.dropout:
            out = self.dropout(out)
        return out

def centre_crop(x, target):
    """
    Center-crop 3D or 4D input tensor along the last two spatial dimensions to match target shape.
    """
    if x is None or target is None:
        return x
    if x.size(2) == target.size(2) and x.size(3) == target.size(3):
        return x  # No cropping needed

    diff_h = x.size(2) - target.size(2)
    diff_w = x.size(3) - target.size(3)

    if diff_h < 0 or diff_w < 0:
        # If x is smaller, interpolate instead of cropping
        return F.interpolate(x, size=(target.size(2), target.size(3)), mode='bilinear', align_corners=False)

    crop_h1 = diff_h // 2
    crop_h2 = diff_h - crop_h1
    crop_w1 = diff_w // 2
    crop_w2 = diff_w - crop_w1

    return x[:, :, crop_h1:x.size(2) - crop_h2, crop_w1:x.size(3) - crop_w2].contiguous()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.block = nn.Sequential(
            self.depthwise_conv,
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            self.pointwise_conv,
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return x + self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.Conv2d(channels, channels // 8, kernel_size=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1, groups=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class ModifiedUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=8):
        super(ModifiedUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(ConvLayer(in_channels, 64, 4, 2, "gn", padding=1), ResidualBlock(64))
        self.enc2 = nn.Sequential(ConvLayer(64, 128, 4, 2, "gn", padding=1), ResidualBlock(128))
        self.enc3 = nn.Sequential(ConvLayer(128, 256, 4, 2, "gn", padding=1), ResidualBlock(256))

        # Bottleneck
        self.bottleneck = nn.Sequential(ConvLayer(256, 512, 3, 1, "gn", padding=1), AttentionBlock(512))

        # Decoder
        self.dec3 = nn.Sequential(ConvLayer(512 + 256, 256, 4, 2, "gn", padding=1, transpose=True), AttentionBlock(256))
        self.dec2 = nn.Sequential(ConvLayer(256 + 128, 128, 4, 2, "gn", padding=1, transpose=True), AttentionBlock(128))
        self.dec1 = nn.Sequential(ConvLayer(128 + 64, 64, 4, 2, "gn", padding=1, transpose=True), AttentionBlock(64))

        # Final layer
        self.final = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self, x):
        x = x[:, :, :512, :] # Crop 513 to 512
        enc1 = checkpoint(self.enc1, x, use_reentrant=False)
        enc2 = checkpoint(self.enc2, enc1, use_reentrant=False)
        enc3 = checkpoint(self.enc3, enc2, use_reentrant=False)
        bottleneck = checkpoint(self.bottleneck, enc3, use_reentrant=False)
        dec3 = checkpoint(self.dec3, torch.cat((bottleneck, enc3), dim=1), use_reentrant=False)
        dec2 = checkpoint(self.dec2, torch.cat((dec3, enc2), dim=1), use_reentrant=False)
        dec1 = checkpoint(self.dec1, torch.cat((dec2, enc1), dim=1), use_reentrant=False)
        out = self.final(dec1)
        out = torch.sigmoid(out)
        out = out.view(-1, 4, 2, *out.shape[2:])
        return F.pad(out, (0, 0, 0, 1))