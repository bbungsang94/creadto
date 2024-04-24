from typing import Dict, Any

import numpy as np
import torch.nn as nn

from creadto.layers.vision import SynthesisPrologue, SynthesisBlock
from creadto.models.mlp import MappingNetwork


class HeadVAE(nn.Module):
    def __init__(self, model_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x) -> Dict[str, Any]:
        pass


class StyleGANAda(nn.Module):
    class SynthesisNetwork(nn.Module):

        def __init__(self, w_dim, img_resolution, img_channels, channel_base=16384, channel_max=512,
                     synthesis_layer='stylegan2'):
            super().__init__()

            self.w_dim = w_dim
            self.img_resolution = img_resolution
            self.img_resolution_log2 = int(np.log2(img_resolution))
            self.img_channels = img_channels
            self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
            self.num_ws = 2 * (len(self.block_resolutions) + 1)
            channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
            self.blocks = nn.ModuleList()
            self.first_block = SynthesisPrologue(channels_dict[self.block_resolutions[0]], w_dim=w_dim,
                                                 resolution=self.block_resolutions[0], img_channels=img_channels,
                                                 synthesis_layer=synthesis_layer)
            for res in self.block_resolutions[1:]:
                in_channels = channels_dict[res // 2]
                out_channels = channels_dict[res]
                block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                       img_channels=img_channels, synthesis_layer=synthesis_layer)
                self.blocks.append(block)

        def forward(self, ws, noise_mode='random'):
            split_ws = [ws[:, 0:2, :]] + [ws[:, 2 * n + 1: 2 * n + 4, :] for n in range(len(self.block_resolutions))]
            x, img = self.first_block(split_ws[0], noise_mode)
            for i in range(len(self.block_resolutions) - 1):
                x, img = self.blocks[i](x, img, split_ws[i + 1], noise_mode)
            return img

    def __init__(self, z_dim, w_dim, w_num_layers, img_resolution, img_channels, channel_max=512, synthesis_layer='stylegan2'):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = _SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, channel_max=channel_max, synthesis_layer=synthesis_layer)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_ws=self.num_ws, num_layers=w_num_layers)

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, noise_mode='random'):
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, noise_mode)
        return img


class _SynthesisNetwork(nn.Module):

    def __init__(self, w_dim, img_resolution, img_channels, channel_base=16384, channel_max=512, synthesis_layer='stylegan2'):
        super().__init__()

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.num_ws = 2 * (len(self.block_resolutions) + 1)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        self.blocks = nn.ModuleList()
        self.first_block = SynthesisPrologue(channels_dict[self.block_resolutions[0]], w_dim=w_dim,
                                             resolution=self.block_resolutions[0], img_channels=img_channels,
                                             synthesis_layer=synthesis_layer)
        for res in self.block_resolutions[1:]:
            in_channels = channels_dict[res // 2]
            out_channels = channels_dict[res]
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, img_channels=img_channels, synthesis_layer=synthesis_layer)
            self.blocks.append(block)

    def forward(self, ws, noise_mode='random'):
        split_ws = [ws[:, 0:2, :]] + [ws[:, 2 * n + 1: 2 * n + 4, :] for n in range(len(self.block_resolutions))]
        x, img = self.first_block(split_ws[0], noise_mode)
        for i in range(len(self.block_resolutions) - 1):
            x, img = self.blocks[i](x, img, split_ws[i + 1], noise_mode)
        return img
