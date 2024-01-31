import numpy as np
import torch


def identity(x):
    return x


def leaky_relu_0_2(x):
    return torch.nn.functional.leaky_relu(x, 0.2)


@torch.jit.script
def clamp_gain(x: torch.Tensor, g: float, c: float):
    return torch.clamp(x * g, -c, c)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulated_conv2d(x, weight, styles, padding=0, demodulate=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOIkk]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


activation_funcs = {
    "linear": {
        "fn": identity,
        "def_gain": 1
    },
    "lrelu": {
        "fn": leaky_relu_0_2,
        "def_gain": np.sqrt(2)
    }
}