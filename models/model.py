# model.py
import torch
import torch.nn as nn

# Define U-Net Architecture with Skip Connections for Multi-Stem Separation
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):  # Set out_channels=4 for the four stems
        super(UNet, self).__init__()

        # Encoder (Downsampling) blocks
        self.enc1 = self.down_block(in_channels, 64)
        self.enc2 = self.down_block(64, 128)
        self.enc3 = self.down_block(128, 256)
        self.enc4 = self.down_block(256, 512)

        # Bottleneck
        self.bottleneck = self.down_block(512, 1024)

        # Decoder (Upsampling) blocks with skip connections
        self.dec4 = self.up_block(1024 + 512, 512)  # Skip connections increase input channels
        self.dec3 = self.up_block(512 + 256, 256)
        self.dec2 = self.up_block(256 + 128, 128)
        self.dec1 = self.up_block(128 + 64, 64)

        # Final output layer
        self.final = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def down_block(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def up_block(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # First encoder layer
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder with skip connections
        dec4 = self.dec4(torch.cat((bottleneck, enc4), dim=1))  # Skip connection from enc4
        dec3 = self.dec3(torch.cat((dec4, enc3), dim=1))  # Skip connection from enc3
        dec2 = self.dec2(torch.cat((dec3, enc2), dim=1))  # Skip connection from enc2
        dec1 = self.dec1(torch.cat((dec2, enc1), dim=1))  # Skip connection from enc1

        # Final output with 4 channels (vocals, drums, bass, other)
        out = self.final(dec1)
        return out

