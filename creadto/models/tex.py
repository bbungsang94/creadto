import copy
from typing import List, Tuple
from PIL.Image import Image
import torch
import numpy as np
import os.path as osp
from torchvision import transforms


class NakedHuman:
    def __init__(self, device="cuda:0"):
        from creadto.models.recon import DetailFaceModel
        model_root = "creadto-model"
        self.flaep = DetailFaceModel()
        self.bridge = np.load(osp.join(model_root, "flame", "flame2smplx_tex_1024.npy"), allow_pickle=True, encoding = 'latin1').item()
        self.head_map = torch.load(osp.join(model_root, "textures", "head_texture_map.pt"))
        self.body_map = torch.load(osp.join(model_root, "textures", "body_texture_map.pt"))
        self.mst = np.load(osp.join(model_root, "textures", "MSTScale", "MSTScaleRGB.npy"))
        self.device = torch.device(device)
        
    def __call__(self, images: List[Image]):
        down_sample = transforms.Compose([transforms.Resize((256, 256))])
        up_sample = transforms.Compose([transforms.Resize((512, 512))])
        crop_images, process = self.flaep.encode_pil(images)
        result = self.flaep.decode(crop_images)
        
        # pre_texture, _ = self.make_head(result["uv_texture_gt"])
        # result = self.flaep.decode(crop_images, external_tex=down_sample(pre_texture))
        output_texture, tone_indices = self.make_head(result["uv_texture_gt"])
        full_texture = self.map_body(output_texture, tone_indices)
        
        return full_texture
    
    def map_body(self, head_albedos: torch.Tensor, tone_indices: torch.Tensor):
        device = head_albedos.device
        dtype = head_albedos.dtype
        for key in self.body_map:
            self.body_map[key] = self.body_map[key].to(device)
        
        x_coords = torch.tensor(self.bridge['x_coords'], dtype=torch.int32, device=device, requires_grad=False)
        y_coords = torch.tensor(self.bridge['y_coords'], dtype=torch.int32, device=device, requires_grad=False)
        target_pixel_ids = torch.tensor(self.bridge['target_pixel_ids'], dtype=torch.int32, device=device, requires_grad=False)
        source_uv_points = torch.tensor(self.bridge['source_uv_points'], dtype=torch.float32, device=device, requires_grad=False)
    
        mst = torch.tensor(self.mst, dtype=dtype, device=device, requires_grad=False) / 255.
        
        body_albedos = []
        for head_albedo, tone in zip(head_albedos, tone_indices):
            body_albedo = copy.deepcopy(self.body_map['default_albedo'])
            base_mask = torch.ones((body_albedo.shape[1], body_albedo.shape[2]), device=mst.device, dtype=torch.bool, requires_grad=False)
            for i in range(3):
                mono_albedo = body_albedo[i]
                min_value, max_value = mst.min(dim=0)[0][i] - (10 / 255.), mst.max(dim=0)[0][i] + (60 / 255.)
                base_mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            
            base_basis = base_mask.unsqueeze(dim=0).expand_as(body_albedo) * (mst[tone] - self.body_map['mean']).view(-1, 1, 1)
            body_albedo = torch.clamp(body_albedo + base_basis, 0.0, 1.0)
            
            source_tex_coords = torch.zeros((source_uv_points.shape[0], source_uv_points.shape[1]), dtype=torch.int32, device=device, requires_grad=False)
            source_tex_coords[:, 0] = torch.clamp(head_albedo.shape[1]*(1.0-source_uv_points[:,1]), 0.0, head_albedo.shape[1]).type(torch.IntTensor)
            source_tex_coords[:, 1] = torch.clamp(head_albedo.shape[2]*(source_uv_points[:,0]), 0.0, head_albedo.shape[2]).type(torch.IntTensor)

            body_albedo[:, y_coords[target_pixel_ids], x_coords[target_pixel_ids]] = head_albedo[:, source_tex_coords[:,0], source_tex_coords[:,1]]
            body_albedos.append(body_albedo)
        
        return body_albedos
            
    def make_head(self, albedo: torch.Tensor):
        """_summary_

        Args:
            albedo (torch.Tensor): (b, 3, 512, 512) albedo images(for head)
        """
        resize = transforms.Compose([transforms.Resize((512, 512))])
        albedo = resize(albedo)

        skin_albedo = albedo * self.head_map['skin_mask'].to(self.device)
        self.save_images("skin-image.jpg", skin_albedo)
        tone_indices, tone_diff = self.detect_skin(skin_albedo=skin_albedo)
        fetched = self.fetch_skin(albedo=albedo, indices=tone_indices, tone_diffs=tone_diff)
        return fetched, tone_indices
    
    def smooth_contour(self, single_image: torch.Tensor):
        """_summary_

        Args:
            single_image (torch.Tensor): "It is NOT a batch tensor."
            contour_mask (torch.Tensor): (512, 512) mask image
        """
        image = single_image.cpu().detach().clone().permute(1, 2, 0)
        contour_mask = self.head_map['contour_mask'][0]
        masked_indices = torch.nonzero(contour_mask.cpu().detach())
        contour_pixels = image[masked_indices[:, 0], masked_indices[:, 1]]
        # 픽셀 값을 무작위로 섞기
        shuffled_indices = torch.randperm(contour_pixels.shape[0])
        shuffled_pixels = contour_pixels[shuffled_indices]
        
        smeared_image = image.clone()
        smeared_image[masked_indices[:, 0], masked_indices[:, 1]] = shuffled_pixels
        smeared_image = smeared_image.permute(2, 0, 1)
        blur = transforms.Compose([transforms.GaussianBlur(kernel_size=21, sigma=(1.5, 2.0))])
        blurred = blur(smeared_image.to(self.head_map['contour_mask'].device)) * self.head_map['contour_mask']
        blurred = blur(single_image) * self.head_map['contour_mask']
        remain = single_image * (1. - self.head_map['contour_mask'])
            
        return blurred + remain
            
    def fetch_skin(self, albedo: torch.Tensor, indices: torch.Tensor, tone_diffs: torch.Tensor) -> torch.Tensor:
        for key in self.head_map:
            self.head_map[key] = self.head_map[key].to(albedo.device)
        mst = torch.tensor(self.mst, dtype=tone_diffs.dtype, device=albedo.device) / 255.
        fetched_result = []
        for skin, index, diff in zip(albedo, indices, tone_diffs):
            face_mask = torch.ones((albedo.shape[2], albedo.shape[3]), device=mst.device, dtype=torch.bool)
            base_mask = torch.ones((self.head_map['default_albedo'].shape[1], self.head_map['default_albedo'].shape[2]), device=mst.device, dtype=torch.bool)
            for i in range(3):
                mono_albedo = skin[i]
                mono_base = self.head_map['default_albedo'][i]
                min_value, max_value = mst.min(dim=0)[0][i] - (10 / 255.), mst.max(dim=0)[0][i] + (60 / 255.)
                face_mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
                base_mask &= (mono_base >= min_value) & (mono_base <= max_value)
            
            face_basis = face_mask.unsqueeze(dim=0).expand_as(skin) * diff.view(-1, 1, 1)
            base_basis = base_mask.unsqueeze(dim=0).expand_as(skin) * (mst[index] - self.head_map['mean']).view(-1, 1, 1)
            fetched_skin = torch.clamp(skin + face_basis, 0.0, 1.0)
            fetched_base = torch.clamp(self.head_map['default_albedo'] + base_basis, 0.0, 1.0)
            # Merge image
            fetched_skin = fetched_skin * self.head_map['face_mask']
            fetched_base = fetched_base * (1. - self.head_map['face_mask'])
            merged_image = fetched_base + fetched_skin
            
            # Blur contour
            result_image = self.smooth_contour(single_image=merged_image)
            fetched_result.append(result_image)
            
        return torch.stack(fetched_result)
                
        
    def detect_skin(self, skin_albedo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            skin_albedo (torch.Tensor): (b, 3, 512, 512) masked-skin images

        Returns:
            skin tone indices (torch.Tensor): (b)
            skin tone difference for rgb (torch.Tensor): (b, 3)
        """
        tone_diffs = torch.zeros((skin_albedo.shape[0], 3), dtype=torch.float32, device=skin_albedo.device)
        tone_indices = torch.zeros((skin_albedo.shape[0]), dtype=torch.int32, device=skin_albedo.device)
        for itr, skin in enumerate(skin_albedo):
            mean_values = torch.zeros(3)
            mask = torch.ones((skin_albedo.shape[2], skin_albedo.shape[3]), device=skin_albedo.device, dtype=torch.bool)
            for i in range(3):
                mono_albedo = skin[i]
                min_value, max_value = self.mst.min(axis=0)[i] - 10, self.mst.max(axis=0)[i] + 60
                min_value, max_value = min_value / 255., max_value / 255.
                mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            
            for i in range(3):
                mono_albedo = skin[i]
                selected_values = mono_albedo[mask]
                if selected_values.numel() > 0:
                    mean_values[i] = selected_values.mean()
                else:
                    mean_values[i] = float('nan')  # 값이 없을 경우 NaN
            # calculated mean value from skin albedo, map real skin to MSTScale
            diff = torch.tensor(self.mst, dtype=mean_values.dtype, device=mean_values.device) / 255 - mean_values 
            tone_index = torch.argmin(abs(diff).sum(dim=1))
            tone_indices[itr] = tone_index
            tone_diffs[itr] = diff[tone_index]
            print("selected mst scale: %02d" % tone_index)
        return tone_indices, tone_diffs
                
    def save_images(self, fname, images: torch.Tensor):
        from torchvision import utils
        for i, image in enumerate(images):
            filename = "%d-th %s" % (i, fname)
            utils.save_image(image, filename)
    
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