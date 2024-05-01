import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image

from creadto.layers import normalize_2nd_moment
from creadto.layers.basic import FullyConnectedLayer


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
