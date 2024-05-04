import torch
import numpy as np
from skimage.transform import estimate_transform, warp


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
    def __init__(self):
        from creadto._external.deca.decalib.deca import DECA
        from creadto._external.deca.decalib.deca import cfg as deca_cfg
        from creadto.models.det import FaceAlignmentLandmarker

        self.detector = FaceAlignmentLandmarker()
        self.reconstructor = DECA(config=deca_cfg)
        self.crop_size = 224
        self.template = np.array([[0, 0], [0, self.crop_size - 1], [self.crop_size - 1, 0]])

    def __call__(self, image: torch.Tensor):
        image = torch.clamp(image * 255., 0, 255)
        image = image.permute(1, 2, 0)
        face_result = self.detector(image)
        tform = estimate_transform('similarity', face_result['points'], self.template)
        image = image / 255.
        image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)[None, ...]
        with torch.no_grad():
            embedding = self.reconstructor.encode(image)
            o, v = self.reconstructor.decode(embedding)
            output = o
            output['latent'] = embedding
            output['visualize'] = v

        return output
