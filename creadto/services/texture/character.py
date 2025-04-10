import os.path as osp
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from creadto.services.reconstruction.head import FLAEP
from creadto.services.segmentation.face import FacialSegmenter
from creadto.utils.decorator import get_instances, service
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images
from creadto.utils.vision import multi_band_blending

@service
class HumanPainter:
    def __init__(self):
        model_root = "creadto-model"
        device = devicex()
        bridge_path = osp.join(model_root, "flame", "flame2smplx_tex_1024.npy")
        albedo_root = osp.join(model_root, "smplx", "high-texture-raw", "white")
        mask_root = osp.join(model_root, "smplx", "high-texture-raw", "masks")
        
        self.bridge = np.load(bridge_path, allow_pickle=True, encoding = 'latin1').item()
        
        resize = transforms.Resize((1024, 1024))
        self.body_albedo = resize(load_image(osp.join(albedo_root, "white_m_8k_raw.png"))[0])
        self.body_albedo = self.body_albedo.type(torch.FloatTensor).to(device)
        self.normal_map = resize(load_image(osp.join(albedo_root, "white_m_8k_normal.png"))[0])
        self.masks = {
            #'iris': resize(load_image(osp.join(mask_root, "weighted_blue_mask.png"), mono=True, integer=False)),
            'iris': torch.zeros_like(self.body_albedo),
            'lips': torch.zeros_like(self.body_albedo),
            'eyelid': load_image(osp.join(mask_root, "face.jpg"), mono=True, integer=False)
        }

        self.segmenter = FacialSegmenter()
        self.flaep = FLAEP()
        
        
    def __call__(self, x):
        segments = self.segmenter(x)
        segments['head_image'] = segments['head_image'] * 255.
        eye_dict = self.segmenter.get_parts_colour(segments,
                                                   [self.segmenter.categories['left_eye'],
                                                    self.segmenter.categories['right_eye']],
                                                   min_thrd=20)
        lip_dict = self.segmenter.get_parts_colour(segments,
                                                   [self.segmenter.categories['lower_lip'],
                                                    self.segmenter.categories['upper_lip']],
                                                   min_thrd=40, max_thrd=255)
        eyebrow_dict = self.segmenter.get_parts_colour(segments,
                                                       [self.segmenter.categories['left_eyebrow'],
                                                        self.segmenter.categories['right_eyebrow']],
                                                       max_thrd=255)
        skin_dict = self.segmenter.get_parts_colour(segments,
                                                    [self.segmenter.categories['skin'],
                                                     self.segmenter.categories['nose']],
                                                    max_thrd=255)
        
        # paint skin
        colored_albedos = self.paint_skin(skin_dict['mean_values'])
        colored_albedos_raw = colored_albedos.detach()
        # displace eyelid
        head_albedos = self.decouple_head_albedo(colored_albedos)
        segments['head_image'] = segments['head_image'] / 255.
        # extract head features
        result = self.flaep(segments['head_image'], external_tex=head_albedos / 255., external_img=skin_dict['enhanced_images'] / 255.)
        up_sample = transforms.Compose([transforms.Resize((512, 512))])
        # dynamic masking from albedo image
        segmented_masks = skin_dict['segmented_masks']
        uv_grid = result['uv_grid']
        albedo_mask = {}
        uv_batch_mask = torch.zeros_like(result["uv_texture_gt"])
        for part_name in ['lower_lip', 'upper_lip', 'mouth', 'left_eyebrow', 'right_eyebrow', 'skin']:
            batch_mask = segmented_masks[:, self.categories[part_name]]
            batch_mask = torch.stack([batch_mask, batch_mask, batch_mask], dim=1)
            partial_mask = F.grid_sample(batch_mask, uv_grid, mode='bilinear', align_corners=False)
            albedo_mask[part_name] = partial_mask
            uv_batch_mask = torch.clamp(uv_batch_mask + partial_mask, 0., 1.)
         # without eye part
        eye_demask = 1. - torch.clamp(self.masks['iris'], 0., 1.)
        eye_demask = eye_demask.to(head_albedos.device)
        # map to body from head
        uv_batch_mask = self.to_body_texture(up_sample(uv_batch_mask).to(head_albedos.device)) * eye_demask
        face_albedos = self.to_body_texture(up_sample(result["uv_texture_gt"]) * 255.)
        
        merged_albedos = (1. - uv_batch_mask) * colored_albedos + uv_batch_mask * face_albedos
        colored_albedos = multi_band_blending(merged_albedos.cpu() / 255., colored_albedos.cpu() / 255., uv_batch_mask.cpu(), levels=6)
        colored_albedos = colored_albedos * 255.
        # paint iris
        colored_albedos = self.paint_with_mask(colored_albedos, self.masks['iris'], eye_dict['mean_values'])
        # paint lips
        colored_albedos = self.paint_with_mask(colored_albedos, self.masks['lips'], lip_dict['mean_values'])
        
        # make normal_map
        head_normal_map = result['uv_detail_normals']
        body_normal_map = self.normal_map.unsqueeze(dim=0).repeat(head_normal_map.shape[0], 1, 1, 1) / 255.
        body_normal_map = body_normal_map.to(head_normal_map.device)
        normal_mask = torch.zeros_like(result["uv_texture_gt"])
        for part_name in ['lower_lip', 'upper_lip', 'mouth', 'left_eyebrow', 'right_eyebrow', 'skin', 'nose']:
            batch_mask = segmented_masks[:, self.categories[part_name]]
            batch_mask = torch.stack([batch_mask, batch_mask, batch_mask], dim=1)
            partial_mask = F.grid_sample(batch_mask, uv_grid, mode='bilinear', align_corners=False)
            normal_mask = torch.clamp(normal_mask + partial_mask, 0., 1.)
        # map to body from head
        normal_mask = self.to_body_texture(up_sample(normal_mask).to(head_albedos.device))
        head_normal_map = self.to_body_texture(up_sample(head_normal_map))
        body_normal_map = (1. - normal_mask) * body_normal_map + normal_mask * head_normal_map
        
        # additional information
        landmarks2d = result['visualize']['landmarks2d'] * result['visualize']['inputs'].shape[-1]
        
        vis_dict = {
            'segmented_images': skin_dict['segmented_images'] / 255.,
            'enhanced_images': skin_dict['enhanced_images'] / 255.,
            'filtered_images': skin_dict['filtered_images'] / 255.,
            'head_images': segments['head_image'],
            'head_albedos': torch.clamp(up_sample(result["uv_texture_gt"]), 0., 1.),
            'head_normal_map': head_normal_map,
            'normal_map': body_normal_map,
            'head_albedos_raw': head_albedos / 255.,
            'full_albedos': colored_albedos,
            'full_albedos_raw': colored_albedos_raw / 255.,
            'landmarks2d': landmarks2d,
            'uv_mask': uv_batch_mask,
            'face_masks': albedo_mask
        }
        
        return vis_dict


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
    return _get_current_(HumanPainter)(image)