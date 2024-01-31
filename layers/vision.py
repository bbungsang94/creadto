import numpy as np
import torch
import torch.nn as nn

from layers import clamp_gain, activation_funcs, identity, modulated_conv2d
from layers.basic import FullyConnectedLayer
from layers.utils import SmoothUpsample


class SynthesisPrologue(torch.nn.Module):

    def __init__(self, out_channels, w_dim, resolution, img_channels, synthesis_layer):
        super().__init__()
        SynthesisLayer = SynthesisLayer2 if synthesis_layer == 'stylegan2' else SynthesisLayer1
        ToRGBLayer = ToRGBLayer2 if synthesis_layer == 'stylegan2' else ToRGBLayer1
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)

    def forward(self, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))
        x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        img = self.torgb(x, next(w_iter))
        return x, img


class SynthesisBlock(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, synthesis_layer):
        super().__init__()
        SynthesisLayer = SynthesisLayer2 if synthesis_layer == 'stylegan2' else SynthesisLayer1
        ToRGBLayer = ToRGBLayer2 if synthesis_layer == 'stylegan2' else ToRGBLayer1
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.resampler = SmoothUpsample()
        self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, resampler=self.resampler)
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)

    def forward(self, x, img, ws, noise_mode):
        w_iter = iter(ws.unbind(dim=1))

        x = self.conv0(x, next(w_iter), noise_mode=noise_mode)
        x = self.conv1(x, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, next(w_iter))
        img = self.resampler(img)
        img = img.add_(y)

        return x, img


class ToRGBLayer2(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        return torch.clamp(x + self.bias[None, :, None, None], -256, 256)


class ToRGBLayer1(nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, _w):
        # note that StyleGAN1's rgb layer doesnt use any style
        w = self.weight * self.weight_gain
        x = nn.functional.conv2d(x, w)
        return torch.clamp(x + self.bias[None, :, None, None], -256, 256)


class SynthesisLayer2(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, resampler=identity, activation='lrelu'):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))

        self.register_buffer('noise_const', torch.randn([resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, padding=self.padding)
        x = self.resampler(x)
        x = x + noise

        return clamp_gain(self.activation(x + self.bias[None, :, None, None]), self.activation_gain * gain, 256 * gain)


class SynthesisLayer1(torch.nn.Module):

    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, resampler=identity, activation='lrelu'):
        super().__init__()
        self.resolution = resolution
        self.resampler = resampler
        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.padding = kernel_size // 2
        self.affine = FullyConnectedLayer(w_dim, out_channels * 2, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.ada_in = AdaIN(out_channels)
        self.register_buffer('noise_const', torch.randn([resolution, resolution]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([1]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode, gain=1):
        styles = self.affine(w)

        noise = None
        if noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        w = self.weight * self.weight_gain
        x = torch.nn.functional.conv2d(x, w, padding=self.padding)

        x = self.resampler(x)
        x = x + noise
        x = clamp_gain(self.activation(x + self.bias[None, :, None, None]), self.activation_gain * gain, 256 * gain)
        x = self.ada_in(x, styles)
        return x


class AdaIN(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(in_channels)

    def forward(self, x, style):
        style = style.unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(x)
        out = gamma * out + beta

        return out