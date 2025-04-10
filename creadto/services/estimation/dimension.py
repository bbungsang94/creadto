from typing import Dict
import torch
import os.path as osp
from creadto.models.legacy import GraphTailor
from creadto.services.classification.gender import GenderClassifier
from creadto.services.reconstruction.body import BLASS
from creadto.services.reconstruction.head import FLAEP
from creadto.utils.decorator import get_instances, service
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images


@service
class DimensionEstimator:
    def __init__(self):
        self.body_reconstructor = BLASS()
        self.face_reconstructor = FLAEP()
        self.gender_classifier = GenderClassifier()
        
        body_guide = torch.load(osp.join('./creadto-model/measure', 'body_dimension_guide.pt'))
        self.body_poses = body_guide['poses']
        self.body_tailor = GraphTailor(body_guide)
        head_guide = torch.load(osp.join('./creadto-model/measure', 'head_dimension_guide.pt'))
        self.head_tailor = GraphTailor(head_guide)
        
    def __call__(self, x):
        with torch.no_grad():
            body = self.body_reconstructor(x)
            face = self.face_reconstructor(x)
            gender = self.gender_classifier(x)
        
        batch_size = body['plane_vertex'].shape[0]
        bodies = dict()
        for pose, value in self.body_poses.items():
            pose_param = value.type(torch.float32)
            pose_param = pose_param.expand(batch_size, pose_param.shape[1], pose_param.shape[2])
            pose_param = pose_param.detach().clone().to(x.device)
            v = self.body_reconstructor.body_decoder.pose_to(body['plane_vertex'], pose_param)

            bodies[pose] = v.detach().clone()
        self.body_tailor.update(bodies)
        body_measurement = self.body_tailor.order(gender=gender, visualize=False, normalize=False)
        body_titles = self.body_tailor.tape
        self.head_tailor.update({'standard': face['plane_verts'].detach().clone()})
        head_measurement = self.head_tailor.order(gender=["female"] * batch_size,
                                                  visualize=False, normalize=False)
        head_titles = self.head_tailor.tape
        full_titles_kor = [x[0] for x in body_titles] + [x[0] for x in head_titles]
        full_measure = torch.concat([body_measurement, head_measurement], dim=1)
        return {'measurement': full_measure, 'titles': full_titles_kor,}
        
     
def _get_current_(cls, **kwargs):
    instance = get_instances(cls)
    if instance == None:
        instance = cls(**kwargs)
    return instance

def estimate_dimension_from_directory(directory: str) -> Dict[str, object]:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return estimate_from_tensor(images)
    
def estimate_dimension_from_file(image_path: str) -> Dict[str, object]:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return estimate_from_tensor(image.unsqueeze(dim=0))

def estimate_from_tensor(image: torch.Tensor) -> Dict[str, object]:
    image = image.to(devicex())
    return _get_current_(DimensionEstimator)(image)