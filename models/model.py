# model.py
import torch
import torch.nn as nn
from models.conv import ConvLayer
from models.crop import centre_crop
from models.resample import Resample1d
from models.layers import AttentionBlock, ResidualBlock

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):  # Set out_channels=4 for the four stems
        super(UNet, self).__init__()

        # Encoder (Downsampling) blocks with ConvLayer and ResidualBlock
        self.enc1 = nn.Sequential(ConvLayer(in_channels, 64, kernel_size=4, stride=2, conv_type="gn"), ResidualBlock(64))
        self.enc2 = nn.Sequential(ConvLayer(64, 128, kernel_size=4, stride=2, conv_type="gn"), ResidualBlock(128))
        self.enc3 = nn.Sequential(ConvLayer(128, 256, kernel_size=4, stride=2, conv_type="gn"), ResidualBlock(256))
        self.enc4 = nn.Sequential(ConvLayer(256, 512, kernel_size=4, stride=2, conv_type="gn"), ResidualBlock(512))

        # Bottleneck with ConvLayer and AttentionBlock
        self.bottleneck = nn.Sequential(ConvLayer(512, 1024, kernel_size=4, stride=2, conv_type="gn"), AttentionBlock(1024))

        # Decoder (Upsampling) blocks with skip connections, ConvLayer, and AttentionBlock
        self.dec4 = nn.Sequential(ConvLayer(1024 + 512, 512, kernel_size=4, stride=2, transpose=True, conv_type="gn"), AttentionBlock(512))
        self.dec3 = nn.Sequential(ConvLayer(512 + 256, 256, kernel_size=4, stride=2, transpose=True, conv_type="gn"), AttentionBlock(256))
        self.dec2 = nn.Sequential(ConvLayer(256 + 128, 128, kernel_size=4, stride=2, transpose=True, conv_type="gn"), AttentionBlock(128))
        self.dec1 = nn.Sequential(ConvLayer(128 + 64, 64, kernel_size=4, stride=2, transpose=True, conv_type="gn"), AttentionBlock(64))

        # Final output layer
        self.final = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder with skip connections and center cropping
        dec4 = self.dec4(torch.cat((bottleneck, centre_crop(enc4, bottleneck)), dim=1))
        dec3 = self.dec3(torch.cat((dec4, centre_crop(enc3, dec4)), dim=1))
        dec2 = self.dec2(torch.cat((dec3, centre_crop(enc2, dec3)), dim=1))
        dec1 = self.dec1(torch.cat((dec2, centre_crop(enc1, dec2)), dim=1))

        # Final output layer
        out = self.final(dec1)
        return out
