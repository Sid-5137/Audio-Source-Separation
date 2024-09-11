import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.conv import ConvLayer
from models.crop import centre_crop
import torch.nn.functional as F
from models.layers import AttentionBlock, ResidualBlock

class ModifiedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(ModifiedUNet, self).__init__()

        # Encoder (Downsampling) blocks
        self.enc1 = nn.Sequential(ConvLayer(in_channels, 64, kernel_size=4, stride=2, padding=1, conv_type="gn"), ResidualBlock(64))
        self.enc2 = nn.Sequential(ConvLayer(64, 128, kernel_size=4, stride=2, padding=1, conv_type="gn"), ResidualBlock(128))
        self.enc3 = nn.Sequential(ConvLayer(128, 256, kernel_size=4, stride=2, padding=1, conv_type="gn"), ResidualBlock(256))
        self.enc4 = nn.Sequential(ConvLayer(256, 512, kernel_size=4, stride=2, padding=1, conv_type="gn"), ResidualBlock(512))

        # Bottleneck with ConvLayer and AttentionBlock
        self.bottleneck = nn.Sequential(ConvLayer(512, 1024, kernel_size=4, stride=2, padding=1, conv_type="gn"), AttentionBlock(1024))

        # Decoder (Upsampling) blocks with adjusted padding
        self.dec4 = nn.Sequential(ConvLayer(1024 + 512, 512, kernel_size=4, stride=2, padding=1, transpose=True, conv_type="gn"), AttentionBlock(512))
        self.dec3 = nn.Sequential(ConvLayer(512 + 256, 256, kernel_size=4, stride=2, padding=1, transpose=True, conv_type="gn"), AttentionBlock(256))
        self.dec2 = nn.Sequential(ConvLayer(256 + 128, 128, kernel_size=4, stride=2, padding=1, transpose=True, conv_type="gn"), AttentionBlock(128))
        self.dec1 = nn.Sequential(ConvLayer(128 + 64, 64, kernel_size=4, stride=2, padding=1, transpose=True, conv_type="gn"), AttentionBlock(64))

        # Final output layer
        self.final = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x=None, target_shape=None, *args, **kwargs):
        # Handle the inputs explicitly
        if x is None:
            if len(args) > 0:
                x = args[0]
            else:
                raise ValueError("Input 'x' is missing. Ensure the input is passed correctly.")

        if target_shape is None:
            target_shape = kwargs.get('target_shape', None)
            if target_shape is None:
                raise ValueError("target_shape must be provided when calling the forward method.")

        # Encoder with skip connections
        enc1 = checkpoint(self.enc1, x)
        enc2 = checkpoint(self.enc2, enc1)
        enc3 = checkpoint(self.enc3, enc2)
        enc4 = checkpoint(self.enc4, enc3)

        # Bottleneck
        bottleneck = checkpoint(self.bottleneck, enc4)

        # Decoder with skip connections and center cropping
        dec4 = checkpoint(self.dec4, torch.cat((bottleneck, centre_crop(enc4, bottleneck)), dim=1))
        dec3 = checkpoint(self.dec3, torch.cat((dec4, centre_crop(enc3, dec4)), dim=1))
        dec2 = checkpoint(self.dec2, torch.cat((dec3, centre_crop(enc2, dec3)), dim=1))
        dec1 = checkpoint(self.dec1, torch.cat((dec2, centre_crop(enc1, dec2)), dim=1))

        # Final output layer
        out = self.final(dec1)
        
        # Resize output to match target shape
        out_resized = F.interpolate(out, size=target_shape[-2:], mode='bilinear', align_corners=False)
        return out_resized
