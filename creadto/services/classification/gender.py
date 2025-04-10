from typing import List

import torch
import torch.nn as nn
import numpy as np

from creadto.services.detection.face import FacialDetector
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images
from creadto.utils.decorator import service, get_instances
from creadto.services.det import GenderClassification

class GenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        device = devicex()
        self.classifier = GenderClassification(device=device)
        self.detector = FacialDetector(device=device)
    
    def forward(self, images):
        crop_images = self.detector(images)       
        result = self.classifier(crop_images)
        return result
        
def _get_current_(cls):
    instance = get_instances(cls)
    if instance == None:
        instance = cls()
    return instance

def classify_from_directory(directory: str) -> List[str]:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return classify_from_tensor(images)
    
def classify_from_file(image_path: str) -> str:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return classify_from_tensor(image.unsqueeze(dim=0))[0]

def classify_from_tensor(image: torch.Tensor) -> List[str]:
    image = image.to(devicex())
    return _get_current_(GenderClassifier)(image)
