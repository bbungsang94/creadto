from typing import Dict
import torch
from torchvision import transforms

from creadto.models.encoder import HighResolutionNet
from creadto.services.det import TokenPoseLandmarker
from creadto.utils.decorator import get_instances, service
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images
from creadto.utils.preprocess.transforms import Crop

# IT IS A MODEL Not a service.
@service
class BodyJointEstimator:
    def __init__(self, device="cpu"):
        weight_pack = torch.load('./creadto-model/app/BLASS-v1-enc-reg-dec-pack')['model']
        # Preprocessor
        self.joint_estimator = TokenPoseLandmarker(device=device)
        self.transformer = {
            'crop': Crop(crop_size=256, is_train=False,
                         scale_dist='normal',
                         scale_factor_max=1.0, scale_factor_min=1.0),
            'normalize': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        }
        self.image_encoder = HighResolutionNet()
        encoder_dict = self.image_encoder.state_dict()
        subset = dict()
        for k, v in weight_pack.items():
            if "backbone." in k:
                subset[k.replace('backbone.', '')] = v
        encoder_dict.update(subset)
        self.image_encoder.load_state_dict(encoder_dict)
        self.image_encoder.to(device)
    
    def __call__(self, x):
        result = dict()
        result['joint_info'] = self.joint_estimator(x)
        result['body_images'] = self._encode(x, result['joint_info'])
        return result
    
    def _encode(self, x, joint_info):
        images = []
        device = x.device
        for stub, joint in zip(x, joint_info):
            center = joint['center']
            center[0], center[1] = int(center[0]), int(center[1])
            cropped, _ = self.transformer['crop'](stub.cpu().detach().numpy(), scale=2.3, center=center)
            normalized = self.transformer['normalize'](cropped / 255.)
            images.append(normalized)
        return torch.stack(images, dim=0).to(device)


def _get_current_(cls, **kwargs):
    instance = get_instances(cls)
    if instance == None:
        instance = cls(**kwargs)
    return instance

def estimate_body_joint_from_directory(directory: str) -> Dict[str, object]:
    images = load_images(directory, integer=False, pad=True)
    images = images.permute(0, 2, 3, 1)
    return estimate_from_tensor(images * 255.)
    
def estimate_body_joint_from_file(image_path: str) -> Dict[str, object]:
    image = load_image(image_path, integer=False).squeeze()
    image = image.permute(1, 2, 0)
    
    return estimate_from_tensor(image.unsqueeze(dim=0) * 255.)

def estimate_from_tensor(image: torch.Tensor) -> Dict[str, object]:
    image = image.to(devicex())
    return _get_current_(BodyJointEstimator, device=devicex())(image)