from dataclasses import dataclass


@dataclass
class LeakyReLU:
    negative_slope: float = 0.01


@dataclass
class ELU:
    alpha: float = 1.0


@dataclass
class PReLU:
    num_parameters: int = 1
    init: float = 0.25

@dataclass
class Activation:
    type: str = 'none'
    inplace: bool = True

    leaky_relu: LeakyReLU = LeakyReLU()
    prelu: PReLU = PReLU()
    elu: ELU = ELU()

@dataclass
class BatchNorm:
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True


@dataclass
class GroupNorm:
    num_groups: int = 32
    eps: float = 1e-05
    affine: bool = True


@dataclass
class LayerNorm:
    eps: float = 1e-05
    elementwise_affine: bool = True


@dataclass
class Normalization:
    type: str = 'none'
    batch_norm: BatchNorm = BatchNorm()
    layer_norm = LayerNorm = LayerNorm()
    group_norm: GroupNorm = GroupNorm()