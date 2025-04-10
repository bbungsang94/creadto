from typing import List
from PIL.Image import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import estimate_transform, warp

class DimensionHuman:
    def __init__(self, head=True):
        female_model = torch.jit.load("./creadto-model/measure/BodyDecoder-f47-10475-v1.pt")
        female_model.eval()
        male_model = torch.jit.load("./creadto-model/measure/BodyDecoder-m47-10475-v1.pt")
        male_model.eval()
        self.models = {
            'body_female': female_model,
            'body_male': male_model
        }
        if head:
            head_model = torch.jit.load("./creadto-model/measure/HeadDecoder-x22-5023-v1.pt")
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
        from creadto.services.det import FaceAlignmentLandmarker

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
        
        output = None
        if True in process:
            output = torch.stack(crop_images, dim=0).to(self.device)
        return output, process
    
    def decode(self, images: torch.Tensor, external_tex=None, external_img=None):
        with torch.no_grad():
            embedding = self.reconstructor.encode(images.to(torch.device("cuda:0")))
            o, v = self.reconstructor.decode(embedding,
                                             external_tex=external_tex,
                                             external_img=external_img)
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

