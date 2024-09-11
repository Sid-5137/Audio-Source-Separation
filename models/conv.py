# conv.py
from torch import nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, padding=0, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.padding = padding

        # Use Conv2d instead of Conv1d
        if self.transpose:
            self.filter = nn.ConvTranspose2d(n_inputs, n_outputs, self.kernel_size, stride, padding=self.padding)
        else:
            self.filter = nn.Conv2d(n_inputs, n_outputs, self.kernel_size, stride, padding=self.padding)

        # Choose normalization type
        NORM_CHANNELS = 8
        if conv_type == "gn":
            assert n_outputs % NORM_CHANNELS == 0
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm2d(n_outputs, momentum=0.01)
        else:
            self.norm = None  # If no normalization is specified

    def forward(self, x):
        # Apply the convolution
        out = self.filter(x)
        if self.norm:
            out = self.norm(out)
        out = F.relu(out) if not self.conv_type == "normal" else F.leaky_relu(out)
        return out
