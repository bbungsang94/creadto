from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from skimage.transform import estimate_transform, warp


class BasicDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=3):
        super().__init__()
        gap = (output_dim - input_dim) // n_layers
        x = input_dim + gap
        y = x + gap
        body = [
            nn.Linear(input_dim, x),
            nn.BatchNorm1d(x),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]
        for _ in range(n_layers - 1):
            layer = [
                nn.Linear(x, y),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
            body += layer
            x = y
            y = x + gap
        self.body = nn.Sequential(*body)
        self.heads = nn.ModuleList()
        for _ in range(3):
            self.heads.append(nn.Linear(x, output_dim))
        self.dim_guide = None

    def forward(self, x) -> Dict[str, Tensor]:
        z = self.body(x)
        o = []
        for head in self.heads:
            # o.append(F.tanh(head(z)))
            o.append(head(z))
        result = {'output': torch.stack(o, dim=2),
                  'latent': z}
        return result


class HeadGATDecoder(nn.Module):
    def __init__(self, n_of_node, node_dim, edge_dim, output_dim, num_heads=5, merge='cat', model_path: str = None):
        from creadto.layers.graph import MultiHeadGATLayer
        super().__init__()
        self.encoder = MultiHeadGATLayer(in_dim=node_dim, out_dim=16, edge_dim=edge_dim,
                                         num_heads=num_heads, merge=merge)
        latent_len = n_of_node * 16 * num_heads
        self.node_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_len, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList()
        for _ in range(3):
            self.heads.append(nn.Linear(4096, output_dim))

    def forward(self, x: Data) -> Dict[str, Any]:
        batch_size = x.num_graphs
        z = self.encoder(x)
        z = self.node_regressor(z.view(batch_size, -1))
        o = []
        for head in self.heads:
            o.append(F.tanh(head(z)))
        result = {'output': torch.stack(o, dim=2),
                  'latent': z}
        return result


class BodyGATDecoder(nn.Module):
    def __init__(self, n_of_node, node_dim, edge_dim, output_dim, num_heads=5, merge='cat'):
        from creadto.layers.graph import MultiHeadGATLayer
        super().__init__()
        self.encoder = MultiHeadGATLayer(in_dim=node_dim, out_dim=16, edge_dim=edge_dim, num_heads=num_heads,
                                         merge=merge)
        latent_len = n_of_node * 16 * num_heads
        self.node_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList()
        for _ in range(3):
            self.heads.append(nn.Linear(8192, output_dim))

    def forward(self, x: Data) -> Dict[str, Any]:
        batch_size = x.num_graphs
        z = self.encoder(x)
        z = self.node_regressor(z.view(batch_size, -1))
        o = []
        for head in self.heads:
            o.append(F.tanh(head(z)))
        result = {'output': torch.stack(o, dim=2),
                  'latent': z}
        return result


class DimensionHuman:
    def __init__(self, head=True):
        female_model = torch.jit.load("./creadto-model/BodyDecoder-f47-10475-v1.pt")
        female_model.eval()
        male_model = torch.jit.load("./creadto-model/BodyDecoder-m47-10475-v1.pt")
        male_model.eval()
        self.models = {
            'body_female': female_model,
            'body_male': male_model
        }
        if head:
            head_model = torch.jit.load("./creadto-model/HeadDecoder-x22-5023-v1.pt")
            head_model.eval()
            self.models['head'] = head_model

    def __call__(self, gender: str, x_body: torch.Tensor, x_head: torch.Tensor = None):
        output = dict()
        with torch.no_grad():
            body_result = self.models['body_' + gender.lower()](x_body)
            body_vertex = body_result['output']
            output['body_vertex'] = body_vertex
            if "head" in self.models and x_head is not None:
                head_result = self.models['head']
                head_vertex = head_result['output']
                output['head_vertex'] = head_vertex
        return output


class DetailFaceModel:
    def __init__(self):
        from creadto._external.deca.decalib.deca import DECA
        from creadto._external.deca.decalib.deca import cfg as deca_cfg
        from creadto.models.det import FaceAlignmentLandmarker

        self.detector = FaceAlignmentLandmarker()
        self.reconstructor = DECA(config=deca_cfg)
        self.crop_size = 224
        self.template = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])

    def __call__(self, image: torch.Tensor):
        image = torch.clamp(image * 255., 0, 255)
        image = image.permute(1, 2, 0)
        face_result = self.detector(image)
        tform = estimate_transform('similarity', face_result['points'], self.template)
        image = image / 255.
        image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)[None, ...]
        with torch.no_grad():
            embedding = self.reconstructor.encode(image)
            o, v = self.reconstructor.decode(embedding)
            output = o
            output['latent'] = embedding
            output['visualize'] = v

        return output
