from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


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
