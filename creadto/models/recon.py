from typing import List
from PIL.Image import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from skimage.transform import estimate_transform, warp


class BLASS:
    def __init__(self):
        from creadto.models.det import TokenPoseLandmarker
        from creadto.networks.encoder import HighResolutionNet
        from creadto.networks.reg import IterativeRegression
        from creadto.layers.basic import MLP
        from creadto._external.smpl.smpl import SMPL
        from creadto.utils.preprocess.transforms import Crop
        # Preprocessor
        self.joint_estimator = TokenPoseLandmarker()
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
        # Processor
        mlp_module = MLP(input_dim=2193, output_dim=145)
        self.body_regressor = IterativeRegression(module=mlp_module)
        self.body_decoder = SMPL("./creadto-model")
        #   image to face
        # Postprocessor
        #   Tailor(body and head)
        #   merge(body head face)
        #   export(obj, glb, fbx)
        self.utils = {
            'smpl_indices':{
                'pose': (0, 132),
                'beta': (132, 142),
                'offset': (142, 146)
            }
        }
        weight_pack = torch.load('./creadto-model/BLASS-v1-enc-reg-dec-pack')['model']
        encoder_dict = self.image_encoder.state_dict()
        subset = dict()
        for k, v in weight_pack.items():
            if "backbone." in k:
                subset[k.replace('backbone.', '')] = v
        encoder_dict.update(subset)
        self.image_encoder.load_state_dict(encoder_dict)
        regressor_dict = self.body_regressor.state_dict()
        subset = dict()
        for k, v in weight_pack.items():
            if "regressor." in k:
                subset[k.replace('regressor.', '')] = v
        regressor_dict.update(subset)
        self.body_regressor.load_state_dict(regressor_dict)

    def eval(self):
        self.image_encoder = self.image_encoder.eval()
        self.body_regressor = self.body_regressor.eval()
        
    def _encode(self, x, joint_info):
        import cv2
        images = []
        for stub, joint in zip(x, joint_info):
            center = joint['center']
            center[0] = int(center[0])
            center[1] = int(center[1])
            cropped, _ = self.transformer['crop'](stub, scale=2.3, center=center)
            normalized = self.transformer['normalize'](cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB) / 255.)
            images.append(normalized)
        return torch.stack(images, dim=0)
    
    def _to_param(self, x):
        with torch.no_grad():
            parameters = dict()
            features = self.image_encoder(x)['concat']
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
                    
            parameters['offset'] = torch.zeros((batch_size, 3), dtype=torch.float32)          
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
        result = dict()
        batch_size = x.shape[0]
        result['joint_info'] = self.joint_estimator(x)
        result['body_images'] = self._encode(x, result['joint_info'])
        result['shape_parameters'] = self._to_param(result['body_images'])
        result['vertex'], result['3d_joint'] = self.body_decoder(**result['shape_parameters'])
        result['plane_vertex'], _ = self.body_decoder(beta=result['shape_parameters']['beta'],
                                                      pose=torch.zeros(batch_size, 55, 3, dtype=torch.float32),
                                                      offset=torch.zeros(batch_size, 3, dtype=torch.float32))
        result['face'] = self.body_decoder.faces
        result['scale'] = result['shape_parameters']['offset'][:, 0].view(-1, 1)
        result['translation'] = result['shape_parameters']['offset'][:, 1:3]
        result['overlay_image'] = np.transpose(np.array(x), [0, 3, 1, 2])
        result['cam_param'] = self.set_render_parameters(result['joint_info'],
                                                         result['translation'],
                                                         result['scale'])
        return result


class DimensionHuman:
    def __init__(self, head=True):
        female_model = torch.jit.load("./creadto-model/BodyDecoder-f47-10475-v1.pt")
        female_model.eval()
        male_model = torch.jit.load("./creadto-model/BodyDecoder-m47-10475-v1.pt")
        male_model.eval()
        self.models = {
            'body_female': female_model,
            'body_male': male_model
        }
        if head:
            head_model = torch.jit.load("./creadto-model/HeadDecoder-x22-5023-v1.pt")
            head_model.eval()
            self.models['head'] = head_model

    def __call__(self, gender: str, x_body: torch.Tensor, x_head: torch.Tensor = None):
        output = dict()
        with torch.no_grad():
            body_result = self.models['body_' + gender.lower()](x_body)
            body_vertex = body_result['output']
            output['body_vertex'] = body_vertex
            if "head" in self.models and x_head is not None:
                head_result = self.models['head']
                head_vertex = head_result['output']
                output['head_vertex'] = head_vertex
        return output


class DetailFaceModel:
    def __init__(self, device="cuda:0"):
        from creadto._external.deca.decalib.deca import DECA
        from creadto._external.deca.decalib.deca import cfg as deca_cfg
        from creadto.models.det import FaceAlignmentLandmarker

        self.detector = FaceAlignmentLandmarker()
        self.reconstructor = DECA(config=deca_cfg, device=device)
        self.crop_size = 224
        self.template = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])
        self.device = torch.device(device)

    def encode(self, images: torch.Tensor):
        crop_images = []
        process = []
        for image in images:
            image = torch.clamp(image * 255., 0, 255)
            image = image.permute(1, 2, 0)
            face_result = self.detector(image)
            if face_result['bbox'] is None:
                process.append(False)
                continue
            tform = estimate_transform('similarity', face_result['points'], self.template)
            image = image / 255.
            image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
            image = image.transpose(2, 0, 1)
            crop_images.append(torch.tensor(image, dtype=torch.float32))
            process.append(True)
        return torch.stack(crop_images, dim=0), process
    
    def encode_pil(self, images: List[Image]):
        crop_images = []
        process = []
        for image in images:
            image = np.asarray(image)
            face_result = self.detector(image)
            if face_result['bbox'] is None:
                process.append(False)
                continue
            tform = estimate_transform('similarity', face_result['points'], self.template)
            image = image / 255.
            image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
            image = image.transpose(2, 0, 1)
            crop_images.append(torch.tensor(image, dtype=torch.float32))
            process.append(True)
        return torch.stack(crop_images, dim=0).to(self.device), process
    
    def decode(self, images: torch.Tensor, external_tex=None):
        with torch.no_grad():
            embedding = self.reconstructor.encode(images.to(torch.device("cuda:0")))
            o, v = self.reconstructor.decode(embedding, external_tex=external_tex)
            output = o
            output['latent'] = embedding
            output['visualize'] = v

        # to cpu
        output['crop_image'] = images
        return output
    
    def __call__(self, image: torch.Tensor):
        crop_images, process = self.encode(image)
        result = self.decode(crop_images)
        result['process'] = process
        return result

