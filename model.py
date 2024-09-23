import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=5, init_features=64):
        super(UNet, self).__init__()
        features = init_features
        
        # Encoder
        self.encoder1 = UNet._block(input_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Bottleneck
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=(2, 2), stride=(2, 2))
        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(2, 2), stride=(2, 2))
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=(2, 2))
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=(2, 2), stride=(2, 2))
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        # Final Convolution
        self.conv = nn.Conv2d(in_channels=features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.crop_and_concat(dec4, enc4)  # Cropping before concatenation
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.crop_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.crop_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.crop_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def crop_and_concat(dec, enc):
        """
        Crops the encoder feature map `enc` to match the shape of the decoder feature map `dec`
        and then concatenates them along the channel axis.
        """
        enc_cropped = UNet.center_crop(enc, dec.shape[2], dec.shape[3])
        return torch.cat([dec, enc_cropped], dim=1)

    @staticmethod
    def center_crop(layer, target_height, target_width):
        """
        Crops the layer to match the target height and width by removing borders.
        """
        _, _, layer_height, layer_width = layer.size()
        delta_height = layer_height - target_height
        delta_width = layer_width - target_width
        crop_top = delta_height // 2
        crop_bottom = delta_height - crop_top
        crop_left = delta_width // 2
        crop_right = delta_width - crop_left

        return layer[:, :, crop_top:layer_height - crop_bottom, crop_left:layer_width - crop_right]
