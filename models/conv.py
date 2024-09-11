# conv.py
from torch import nn
from torch.nn import functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, conv_type="normal", transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        NORM_CHANNELS = 8  # Number of channels for GroupNorm

        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)

        if conv_type == "gn":
            assert out_channels % NORM_CHANNELS == 0
            self.norm = nn.GroupNorm(out_channels // NORM_CHANNELS, out_channels)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        return F.relu(x)
