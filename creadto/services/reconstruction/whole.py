from typing import Dict
import torch
from creadto.models.legacy import ModelConcatenator
# from creadto.services.reconstruction.body import BLASS
from creadto.services.reconstruction.head import FLAEP
from creadto.utils.decorator import get_instances, service
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images

@service
class HLAMP:
    def __init__(self):
        # self.body_reconstructor = BLASS()
        self.face_reconstructor = FLAEP()
        self.concatenator = ModelConcatenator(root="./creadto-model/smplx")
        
    def __call__(self, x):
        with torch.no_grad():
            body = self.body_reconstructor(x)
            face = self.face_reconstructor(x)
        humans = self.concatenator.update_model(body=body['plane_vertex'],
                                                head=face['plane_verts'],
                                                visualize=False)
        return {'vertex': humans['model']['body'],
                'face': body['face']}
     
def _get_current_(cls, **kwargs):
    instance = get_instances(cls)
    if instance == None:
        instance = cls(**kwargs)
    return instance

def reconstruct_human_from_directory(directory: str) -> Dict[str, object]:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return reconstruct_from_tensor(images)
    
def reconstruct_human_from_file(image_path: str) -> Dict[str, object]:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return reconstruct_from_tensor(image.unsqueeze(dim=0))

def reconstruct_from_tensor(image: torch.Tensor) -> Dict[str, object]:
    image = image.to(devicex())
    return _get_current_(HLAMP)(image)