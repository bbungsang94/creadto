import torch
import torch.nn as nn
from torchvision.transforms import transforms


class BasicFacialLandmarker:
    def __init__(self, model):
        self.model = model.cpu()
        self.n_markers = 478
        self.trans = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
        ])

    def to(self, device):
        self.model.to(device)

    def __call__(self, x):
        x = self.trans(x)
        latent, o = self.model(x)
        result = {'output': o,
                  'latent': latent}
        return result
