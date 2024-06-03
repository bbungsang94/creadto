import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image

from creadto.layers import normalize_2nd_moment
from creadto.layers.basic import FullyConnectedLayer


class IterativeRegression(nn.Module):
    def __init__(
        self,
        module,
        mean_param = None,
        num_stages: int = 3,
        append_params: bool = True,
        learn_mean: bool = False,
        detach_mean: bool = False,
        dim: int = 1,
    ) -> None:
        super(IterativeRegression, self).__init__()

        self.module = module
        self._num_stages = num_stages
        self.dim = dim
        self.append_params = append_params
        self.detach_mean = detach_mean
        self.learn_mean = learn_mean
        if mean_param is None:
            mean_param = torch.load('./creadto-model/body_iterative_regressor_mean.pth')
        if learn_mean:
            self.register_parameter(
                'mean_param', nn.Parameter(mean_param, requires_grad=True))
        else:
            self.register_buffer('mean_param', mean_param)

    def get_mean(self):
        return self.mean_param.clone()

    @property
    def num_stages(self):
        return self._num_stages

    def extra_repr(self):
        msg = [
            f'Num stages = {self.num_stages}',
            f'Concatenation dimension: {self.dim}',
            f'Detach mean: {self.detach_mean}',
            f'Learn mean: {self.learn_mean}',
        ]
        return '\n'.join(msg)

    def forward(self, features, cond = None):
        ''' Computes deltas on top of condition iteratively

            Parameters
            ----------
                features: torch.Tensor
                    Input features computed by a NN backbone
                cond: Condition vector used as the initial point of the
                    iterative regression method.

            Returns
            -------
                parameters: List[torch.Tensor]
                    A list of tensors, where each element corresponds to a
                    different stage
                deltas: List[torch.Tensor]
                    A list of tensors, where each element corresponds to a
                    the estimated offset at each stage
        '''
        batch_size = features.shape[0]
        expand_shape = [batch_size] + [-1] * len(features.shape[1:])

        parameters = []
        deltas = []
        module_input = features

        if cond is None:
            cond = self.mean_param.expand(*expand_shape).clone()

        # Detach mean
        if self.detach_mean:
            cond = cond.detach()

        if self.append_params:
            assert features is not None, (
                'Features are none even though append_params is True')

            module_input = torch.cat([
                module_input,
                cond],
                dim=self.dim)
        deltas.append(self.module(module_input))
        num_params = deltas[-1].shape[1]
        parameters.append(cond[:, :num_params].clone() + deltas[-1])

        for stage_idx in range(1, self.num_stages):
            module_input = torch.cat(
                [features, parameters[stage_idx - 1]], dim=-1)
            params_upd = self.module(module_input)
            parameters.append(parameters[stage_idx - 1] + params_upd)

        return parameters, deltas
    
    
class HumanDimEstimator:
    def __init__(self, device="cuda:0"):
        from body_matrix import load, measure
        from creadto.models.det import TokenPoseLandmarker, MediaPipeLandmarker

        class FakeTransform:
            def __init__(self):
                self.last = None

            def __call__(self, x, *args, **kwargs):
                self.last = np.array(x, *args, **kwargs)
                return self

            def to(self, *args, **kwargs):
                return self.last

        self.facial_model = MediaPipeLandmarker('./creadto-model/mp_face_landmarker.task', 0.5, 0.5)
        self.kp_model, self.kp_transform = TokenPoseLandmarker(device), FakeTransform()
        self.sg_model, self.sg_transform = load.segment_model(device)
        self.device = device
        self.collate_fn = measure.find_real_measures

    def __call__(self, x: Image):
        measurement = dict()
        height, leg, hip, shoulder, markers, keypoints = self.collate_fn(
            image_frame=x,
            device=self.device,
            keypoints_model=self.kp_model,
            keypoints_transform=self.kp_transform,
            segment_model=self.sg_model,
            segment_transform=self.sg_transform
        )
        markers.update(keypoints)
        head_height = self.distance(markers['top_head'], markers['chin'])
        measurement['Stature'] = height
        measurement['Crotch Height'] = leg
        measurement['Hip Width'] = hip
        measurement['Shoulder to neck(full)'] = shoulder
        measurement['Head Height'] = 22. * head_height
        return measurement

    @staticmethod
    def distance(a, b):
        x_dif = (a[0] - b[0])
        y_dif = (a[1] - b[1])
        dif = math.sqrt(x_dif * x_dif + y_dif * y_dif)
        return dif


class BasicRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x) -> Dict[str, Any]:
        o = self.encoder(x)
        result = {'output': o * 2,
                  'latent': o}
        return result


class MappingNetwork(nn.Module):

    def __init__(self, z_dim, w_dim, num_ws=None, num_layers=8, activation='lrelu', lr_multiplier=0.01,
                 w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        if num_ws is None:
            self.w_avg_beta = None
        else:
            self.w_avg_beta = w_avg_beta
        embed_features = 0

        # features_list = [z_dim ] + [w_dim] * num_layers
        features_list = [z_dim + embed_features] + [w_dim] * num_layers

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            self.layers.append(FullyConnectedLayer(in_features, out_features,
                                                   activation=activation, lr_multiplier=lr_multiplier))

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None

        if self.z_dim > 0:
            x = normalize_2nd_moment(z)

        # Main layers.
        for idx in range(self.num_layers):
            x = self.layers[idx](x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x
