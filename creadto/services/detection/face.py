from typing import List

import torch
import numpy as np
from skimage.transform import estimate_transform, warp

from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images
from creadto.utils.decorator import service, get_instances
from creadto.services.det import FaceAlignmentLandmarker

# Is it the service?
@service
class FacialDetector:
    def __init__(self, device="cpu", crop_size=224):
        self.facial_detector = FaceAlignmentLandmarker(device=device)
        self.crop_size = crop_size
        self.template = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
    
    def __call__(self, images):
        crop_images = []
        device = images.device
        for image in images:
            face_result = self.facial_detector(image)
            tform = estimate_transform('similarity', face_result['points'], self.template)

            image = image / 255.
            image = warp(image.cpu(), tform.inverse, output_shape=(self.crop_size, self.crop_size))
            image = image.transpose(2, 0, 1)
            crop_images.append(torch.tensor(image))
            
        return torch.stack(crop_images, dim=0).to(device)
        
def _get_current_(cls, **kwargs):
    instance = get_instances(cls)
    if instance == None:
        instance = cls(**kwargs)
    return instance

def detect_face_from_directory(directory: str) -> torch.Tensor:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return detect_from_tensor(images)
    
def detect_face_from_file(image_path: str) -> torch.Tensor:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return detect_from_tensor(image.unsqueeze(dim=0))[0]

def detect_from_tensor(image: torch.Tensor) -> torch.Tensor:
    device = devicex()
    image = image.to(device)
    return _get_current_(FacialDetector, crop_size=224, device=device)(image)