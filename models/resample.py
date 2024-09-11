# resample.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect"):
        super(Resample1d, self).__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels
        cutoff = 0.5 / stride
        filter = self.build_sinc_filter(kernel_size, cutoff)
        self.filter = nn.Parameter(torch.from_numpy(np.repeat(filter[None, :], channels, axis=0)), requires_grad=False)

    def forward(self, x):
        if self.padding != "valid":
            num_pad = (self.kernel_size - 1) // 2
            x = F.pad(x, (num_pad, num_pad), mode=self.padding)

        if self.transpose:
            x = F.conv_transpose1d(x, self.filter, stride=self.stride, groups=self.channels)
        else:
            x = F.conv1d(x, self.filter, stride=self.stride, groups=self.channels)
        return x

    @staticmethod
    def build_sinc_filter(kernel_size, cutoff):
        M = kernel_size - 1
        filter = np.sinc(2 * cutoff * (np.arange(kernel_size) - M / 2))
        window = np.hanning(kernel_size)
        return filter * window / np.sum(filter * window)
