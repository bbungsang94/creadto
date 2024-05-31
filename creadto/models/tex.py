from typing import List
from PIL.Image import Image
import torch
import numpy as np
import os.path as osp


class NakedHuman:
    def __init__(self):
        from creadto.models.recon import DetailFaceModel
        model_root = "creadto-model"
        self.flaep = DetailFaceModel()
        self.head_map = torch.load(osp.join(model_root, "textures", "head_texture_map.pt"))
        self.mst = np.load(osp.join(model_root, "textures", "MSTScale", "MSTScale.npy"))
    
    def __call__(self, images: List[Image]):
        crop_images, process = self.flaep.encode_pil(images)
        result = self.flaep.decode(crop_images)
        
        image_texture = result["uv_texture_gt"]
    
    def make_head(self, albedo: torch.Tensor):
        """_summary_

        Args:
            albedo (torch.Tensor): (b, 3, 512, 512) albedo images(for head)
        """
        skin_albedo = albedo * self.head_map['skin_mask']
        for i in range(3):
            

if __name__ == "__main__":
    import torchvision.io as io
    from torchvision.transforms import ToTensor
    import cv2
    
    model_root = "creadto-model"
    albedo_path = osp.join(model_root, "flame", "default_texture", "realistic_origin.png")
    skin_mask_path = osp.join(model_root, "flame", "mask_images", "skin.jpg")
    observed_mask_path = osp.join(model_root, "flame", "mask_images", "observed_mask.jpg")
    contour_path = osp.join(model_root, "flame", "mask_images", "face_contour.jpg")
    default_albedo = cv2.imread(albedo_path)
    default_albedo = cv2.resize(default_albedo, (512, 512))
    default_albedo = cv2.cvtColor(default_albedo, cv2.COLOR_BGR2RGB)
    skin_mask = cv2.imread(skin_mask_path, cv2.IMREAD_GRAYSCALE)
    face_mask = cv2.imread(observed_mask_path, cv2.IMREAD_GRAYSCALE)
    contour_mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
    _, skin_mask = cv2.threshold(skin_mask, 128, 255, cv2.THRESH_BINARY)
    _, face_mask = cv2.threshold(face_mask, 128, 255, cv2.THRESH_BINARY)
    _, contour_mask = cv2.threshold(contour_mask, 128, 255, cv2.THRESH_BINARY)
    default_skin_image = cv2.bitwise_and(default_albedo, default_albedo, mask=skin_mask)
    trans = ToTensor()
    skin_mask = trans(skin_mask)
    face_mask = trans(face_mask)
    contour_mask = trans(contour_mask)
    default_albedo = trans(default_albedo)
    print(torch.unique(skin_mask[0]))
    print(torch.unique(face_mask[0]))
    print(torch.unique(contour_mask[0]))
    head_texture_kit = {
        'default_albedo': default_albedo,
        'mean': torch.tensor([0.8115, 0.7306, 0.6998], dtype=torch.float32),
        'skin_mask': skin_mask.expand_as(default_albedo),
        'face_mask': face_mask.expand_as(default_albedo),
        'contour_mask': contour_mask.expand_as(default_albedo),
    }
    torch.save(head_texture_kit, "head_texture_map.pt")