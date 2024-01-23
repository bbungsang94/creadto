from typing import Dict, Any

import torch.nn as nn


class FlameDecoder(nn.Module):
    def __init__(self, model_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x) -> Dict[str, Any]:
        pass
