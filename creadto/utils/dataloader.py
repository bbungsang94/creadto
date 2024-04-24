import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


class GraphLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        """
        :param dataset:
        :param kwargs:
        """
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device

    def _collate_fn(self, batch):
        graphs = []
        labels = []
        for parameter in batch:
            x, y = parameter
            graphs.append(x)
            labels.append(y)
        graphs = Batch.from_data_list(graphs)
        labels = torch.stack(labels)

        return graphs.to(self.device), labels.to(self.device)


class TensorLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        """
        :param dataset:
        :param kwargs:
        """
        super().__init__(dataset, **kwargs)
        self.collate_fn = self._collate_fn
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device

    def _collate_fn(self, batch):
        x = []
        y = []
        for parameter in batch:
            sample, label = parameter
            x.append(sample)
            y.append(label)
        x = torch.stack(x).to(self.device)
        y = torch.stack(y).to(self.device)

        return x, y
