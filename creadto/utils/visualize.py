import networkx as nx
import torch_geometric
from torch_geometric.data import Data


def draw_graph(data: Data, unidirectional=False):
    g = torch_geometric.utils.to_networkx(data, to_undirected=unidirectional)
    nx.draw(g, with_labels=True)