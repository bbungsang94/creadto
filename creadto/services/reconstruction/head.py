from typing import Dict

import torch

from creadto.services.detection.face import FacialDetector
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images
from creadto.utils.decorator import service, get_instances
from creadto._external.deca.decalib.deca import DECA
from creadto._external.deca.decalib.deca import cfg as deca_cfg

# Is it the service?
@service
class FLAEP:
    def __init__(self):
        device = devicex()
        self.detector = FacialDetector(device=device, crop_size=224)
        self.reconstructor = DECA(config=deca_cfg, device=device)
    
    def __call__(self, images, external_tex=None, external_img=None):
        crop_images = self.detector(images)
        with torch.no_grad():
            embedding = self.reconstructor.encode(crop_images)
            o, v = self.reconstructor.decode(embedding,
                                             external_tex=external_tex,
                                             external_img=external_img)
        o.update(v)
        return o
    
def _get_current_(cls):
    instance = get_instances(cls)
    if instance == None:
        instance = cls()
    return instance

def reconstruct_head_from_directory(directory: str) -> Dict[str, object]:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return reconstruct_from_tensor(images)
    
def reconstruct_head_from_file(image_path: str) -> Dict[str, object]:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return reconstruct_from_tensor(image.unsqueeze(dim=0))

def reconstruct_from_tensor(image: torch.Tensor) -> Dict[str, object]:
    image = image.to(devicex())
    return _get_current_(FLAEP)(image)