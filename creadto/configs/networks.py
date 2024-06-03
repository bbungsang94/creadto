from typing import Tuple
from dataclasses import dataclass

from creadto.configs.activations import Activation, Normalization


@dataclass
class HRNet:
    @dataclass
    class Stage:
        num_modules: int = 1
        num_branches: int = 1
        num_blocks: Tuple[int] = (4,)
        num_channels: Tuple[int] = (64,)
        block: str = 'BOTTLENECK'
        fuse_method: str = 'SUM'

    @dataclass
    class SubSample:
        num_layers: int = 3
        num_filters: Tuple[int] = (512,) * num_layers
        kernel_size: int = 3
        norm_type: str = 'bn'
        activ_type: str = 'relu'
        dim: int = 2
        kernel_sizes = [kernel_size] * len(num_filters)
        stride: int = 2
        strides: Tuple[int] = (stride,) * len(num_filters)
        padding: int = 1

    use_old_impl: bool = False
    pretrained_layers: Tuple[str] = ('*',)
    pretrained_path: str = (
        './creadto-model/hrnet-4stage-basic'
    )
    stage1: Stage = Stage()
    stage2: Stage = Stage(num_branches=2, num_blocks=(4, 4),
                          num_channels=(48, 96), block='BASIC')
    stage3: Stage = Stage(num_modules=4, num_branches=3,
                          num_blocks=(4, 4, 4),
                          num_channels=(48, 96, 192),
                          block='BASIC')
    stage4: Stage = Stage(num_modules=3, num_branches=4,
                          num_blocks=(4, 4, 4, 4,),
                          num_channels=(48, 96, 192, 384),
                          block='BASIC',
                          )


@dataclass
class MLP:
    layers: Tuple[int] = (1024, 1024)
    activation: Activation = Activation()
    normalization: Normalization = Normalization()
    preactivated: bool = False
    dropout: float = 0.5
    init_type: str = 'xavier'
    gain: float = 0.01
    bias_init: float = 0.0
