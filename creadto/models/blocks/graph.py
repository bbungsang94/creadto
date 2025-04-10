import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=5, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATConv(in_channels=in_dim, out_channels=out_dim, edge_dim=edge_dim))
        self.merge = merge

    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_attr
        head_outs = [attn_head(x, edge_index, edge_attr=edge_feature) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))