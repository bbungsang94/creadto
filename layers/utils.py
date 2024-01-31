import torch
import torch.nn as nn


class SmoothDownsample(nn.Module):

    def __init__(self):
        super().__init__()
        kernel = [[1, 3, 3, 1],
                  [3, 9, 9, 3],
                  [3, 9, 9, 3],
                  [1, 3, 3, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d((2, 1, 2, 1))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = nn.functional.conv2d(x, self.kernel).view(b, c, h, w)
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='nearest', recompute_scale_factor=False)
        return x


class SmoothUpsample(nn.Module):

    def __init__(self):
        super().__init__()
        kernel = [[1, 3, 3, 1],
                  [3, 9, 9, 3],
                  [3, 9, 9, 3],
                  [1, 3, 3, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d((2, 1, 2, 1))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.pad(x)
        x = nn.functional.conv2d(x, self.kernel).view(b, c, h * 2, w * 2)
        return x
