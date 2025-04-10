from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

from creadto.models.blocks.layers.basic import MLP
from creadto.models.reg import IterativeRegression
from creadto._external.smpl.smpl import SMPL
from creadto.services.estimation.joint import BodyJointEstimator
from creadto.utils.decorator import get_instances, service
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images

@service
class BLASS:
    def __init__(self):
        device = devicex()
        weight_pack = torch.load('./creadto-model/app/BLASS-v1-enc-reg-dec-pack')['model']
        # Preprocessor
        self.joint_estimator = BodyJointEstimator(device=device)
        # Processor
        self.body_decoder = SMPL("./creadto-model/smplx")

        self.utils = {
            'smpl_indices':{
                'pose': (0, 132),
                'beta': (132, 142),
                'offset': (142, 146)
            }
        }
        self.body_regressor = IterativeRegression(module=MLP(input_dim=2193, output_dim=145))
        regressor_dict = self.body_regressor.state_dict()
        subset = dict()
        for k, v in weight_pack.items():
            if "regressor." in k:
                subset[k.replace('regressor.', '')] = v
        regressor_dict.update(subset)
        self.body_regressor.load_state_dict(regressor_dict)
        self.body_regressor.to(device), self.body_decoder.to(device)

    def eval(self):
        self.image_encoder = self.image_encoder.eval()
        self.body_regressor = self.body_regressor.eval()
    
    def _to_param(self, x):
        with torch.no_grad():
            parameters = dict()
            features = self.joint_estimator.image_encoder(x)['concat']
            param_space, _ = self.body_regressor(features)
            param_space = param_space[-1]
            batch_size = param_space.shape[0]
            for key, value in self.utils['smpl_indices'].items():
                parameters[key] = param_space[:, value[0]:value[1]]
                if key == 'pose':
                    parameters[key] = torch.zeros((batch_size, 22, 3, 3))
                    for i, pose_val in enumerate(param_space[:, value[0]:value[1]]):
                        pose_val = pose_val.view(-1, 3, 2)
                        b1 = F.normalize(pose_val[:, :, 0].clone(), dim=1)
                        dot_prod = torch.sum(b1 * pose_val[:, :, 1].clone(), dim=1, keepdim=True)
                        b2 = F.normalize(pose_val[:, :, 1] - dot_prod * b1, dim=1)
                        b3 = torch.cross(b1, b2, dim=1)
                        rot_mats = torch.stack([b1, b2, b3], dim=-1)

                        pose_mat = rot_mats.view(1, -1, 3, 3)
                        parameters[key][i] = pose_mat
                    bumper = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).expand(batch_size, 33, -1, -1).contiguous()
                    parameters[key] = torch.cat([parameters[key], bumper], dim=1)
                    parameters['pose'][:, 0, :] = torch.eye(3, dtype=torch.float32)
                parameters[key] = parameters[key].to(x.device)
            parameters['offset'] = torch.zeros((batch_size, 3), dtype=torch.float32, device=x.device)          
        return parameters
    
    def set_render_parameters(self, joint_info, cam_trans, cam_scale):
        h, w = 450, 450
        parameters = {'shift_x': [], 'shift_y': [],
                      'transl': [],
                      'focal_length_in_mm': [], 'focal_length_in_px': [],
                      'center': [], 'sensor_width': [],}
        for i, info in enumerate(joint_info):
            bbox, center = info['boxes'][0], info['center'][0:2]
            box_size = max(bbox[2:4])
            z = 2 * 500 / (cam_scale[i] * box_size)

            transl = [cam_trans[i, 0].item(), cam_trans[i, 1].item(), z.item()]
            shift_x = - (center[0] / w - 0.5)
            shift_y = (center[1] - 0.5 * h) / w
            focal_length_in_mm = 5000 / w * 23
            parameters['shift_x'].append(shift_x)
            parameters['shift_y'].append(shift_y)
            parameters['transl'].append(transl)
            parameters['focal_length_in_mm'].append(focal_length_in_mm)
            parameters['focal_length_in_px'].append(5000)
            parameters['center'].append(center)
            parameters['sensor_width'].append(23)

        return parameters
        
    def __call__(self, x):
        batch_size = x.shape[0]
        device = x.device
        result = self.joint_estimator(x)
        result['shape_parameters'] = self._to_param(result['body_images'])
        result['vertex'], result['3d_joint'] = self.body_decoder(**result['shape_parameters'])
        result['plane_vertex'], _ = self.body_decoder(beta=result['shape_parameters']['beta'],
                                                      pose=torch.zeros(batch_size, 55, 3, dtype=torch.float32, device=device),
                                                      offset=torch.zeros(batch_size, 3, dtype=torch.float32, device=device))
        result['face'] = self.body_decoder.faces
        result['scale'] = result['shape_parameters']['offset'][:, 0].view(-1, 1)
        result['translation'] = result['shape_parameters']['offset'][:, 1:3]
        result['cam_param'] = self.set_render_parameters(result['joint_info'],
                                                         result['translation'],
                                                         result['scale'])
        return result
    
def _get_current_(cls, **kwargs):
    instance = get_instances(cls)
    if instance == None:
        instance = cls(**kwargs)
    return instance

def reconstruct_body_from_directory(directory: str) -> Dict[str, object]:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return reconstruct_from_tensor(images)
    
def reconstruct_body_from_file(image_path: str) -> Dict[str, object]:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return reconstruct_from_tensor(image.unsqueeze(dim=0))

def reconstruct_from_tensor(image: torch.Tensor) -> Dict[str, object]:
    image = image.to(devicex())
    return _get_current_(BLASS)(image)