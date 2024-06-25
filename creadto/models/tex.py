import copy
import facer
from typing import Dict, List, Tuple
from PIL.Image import Image
import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
from torchvision import transforms


class PaintHuman:
    def __init__(self, device="cuda:0"):
        from creadto.utils.io import load_image
        from creadto.models.recon import DetailFaceModel
        model_root = "creadto-model"
        self.flaep = DetailFaceModel()
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device,
                                                 model_path=osp.join(model_root, "mobilenet0.25_Final.pth"))
        self.face_parser = facer.face_parser('farl/celebm/448', device=device,
                                             model_path=osp.join(model_root, "face_parsing.farl.celebm.main_ema_181500_jit.pt")) # optional "farl/lapa/448"
        self.bridge = np.load(osp.join(model_root, "flame", "flame2smplx_tex_1024.npy"), allow_pickle=True, encoding = 'latin1').item()
        
        mask_root = osp.join(model_root, "textures", "MSTScale", "Samples")
        self.body_albedo = load_image(osp.join(mask_root, "mono_body-masks.png"))[0]
        self.body_albedo = self.body_albedo.type(torch.FloatTensor).to(device)
        self.masks = {
            'iris': load_image(osp.join(mask_root, "weighted_green_mask.png"), mono=True, integer=False),
            'lips': load_image(osp.join(mask_root, "weighted_red_mask.png"), mono=True, integer=False),
            'eyelid': load_image(osp.join(mask_root, "face.jpg"), mono=True, integer=False)
        }
        self.categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3, 
                           "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                           "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                           "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                           "hat": 16, "earring": 17, "necklace": 18}
    def __call__(self, images: List[Image]):
        # extract head images
        head_images, process = self.flaep.encode_pil(images)
        # face parsing
        head_images = head_images * 255.
        eye_dict = self.get_parts_colour(head_images, [self.categories['left_eye'], self.categories['right_eye']], min_thrd=20)
        lip_dict = self.get_parts_colour(head_images, [self.categories['lower_lip'], self.categories['upper_lip']], min_thrd=40, max_thrd=255)
        eyebrow_dict = self.get_parts_colour(head_images, [self.categories['left_eyebrow'], self.categories['right_eyebrow']], max_thrd=255)
        skin_dict = self.get_parts_colour(head_images, [self.categories['skin'], self.categories['nose']], max_thrd=255)
        # paint skin
        colored_albedos = self.paint_skin(skin_dict['mean_values'])
        # paint iris
        colored_albedos = self.paint_with_mask(colored_albedos, self.masks['iris'], eye_dict['mean_values'])
        # paint lips
        colored_albedos = self.paint_with_mask(colored_albedos, self.masks['lips'], lip_dict['mean_values'])
        # displace eyelid
        head_albedos = self.decouple_head_albedo(colored_albedos)
        head_images = head_images  / 255.
        # extract head features
        result = self.flaep.decode(head_images, external_tex=head_albedos / 255., external_img=skin_dict['enhanced_images'] / 255.)
        up_sample = transforms.Compose([transforms.Resize((512, 512))])
        # dynamic masking from albedo image
        segmented_masks = skin_dict['segmented_masks']
        uv_grid = result['uv_grid']
        albedo_mask = {}
        uv_batch_mask = torch.zeros_like(result["uv_texture_gt"])
        for part_name in ['skin', 'lower_lip', 'upper_lip', 'mouth', 'nose', 'left_eyebrow', 'right_eyebrow']:
            batch_mask = segmented_masks[:, self.categories[part_name]]
            batch_mask = torch.stack([batch_mask, batch_mask, batch_mask], dim=1)
            partial_mask = F.grid_sample(batch_mask, uv_grid, mode='bilinear', align_corners=False)
            albedo_mask[part_name] = partial_mask
            uv_batch_mask = torch.clamp(uv_batch_mask + partial_mask, 0., 1.)
        
        # map to body from head
        uv_batch_mask = self.to_body_texture(up_sample(uv_batch_mask).to(head_albedos.device))
        face_albedos = self.to_body_texture(up_sample(result["uv_texture_gt"]) * 255.)
        colored_albedos = (1. - uv_batch_mask) * colored_albedos + uv_batch_mask * face_albedos
        # fetch eyebrow
        
        # additional information
        landmarks2d = result['visualize']['landmarks2d'] * result['visualize']['inputs'].shape[-1]
        
        vis_dict = {
            'segmented_images': skin_dict['segmented_images'] / 255.,
            'enhanced_images': skin_dict['enhanced_images'] / 255.,
            'filtered_images': skin_dict['filtered_images'] / 255.,
            'head_images': head_images,
            'face_detection': process,
            'head_albedos': torch.clamp(up_sample(result["uv_texture_gt"]), 0., 1.),
            'full_albedos': colored_albedos,
            'landmarks2d': landmarks2d,
            'uv_mask': uv_batch_mask,
            'face_masks': albedo_mask
        }
        
        return vis_dict

    def get_parts_colour(self, images: torch.Tensor, parts: List[int], min_thrd = 5, max_thrd = 170) -> Dict[str, object]:
        """From head image (b, 3, 224, 224), extract eye(iris) colour without sclera.

        Args:
            images (torch.Tensor): (b, 3, 224, 224) head image [0, 255]
            min_thrd (int, optional): for mask of sclera. Defaults to 0.
            max_thrd (int, optional): _for mask of sclera. Defaults to 170.

        Returns:
            Dict[str, object]: Dict of operation results such as input images, processed images, colour values, and file names
        """
        
        device = self.body_albedo.device
        
        # Need refactoring part!, we can get processed arguments
        with torch.inference_mode():
            faces = self.face_detector(images)
            faces = self.face_parser(images, faces)
        vis_images = facer.draw_bchw(images, faces)
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)
        
        masks = torch.zeros((seg_probs[:, 0].shape), device=seg_probs.device, dtype=seg_probs.dtype)
        for part in parts:
            masks += seg_probs[:, part]
        mean_values = []
        enhanced_images = []
        filtered_images = []
        for image, mask in zip(images, masks):
            vis = image * mask.expand_as(image)
            
            filter = torch.ones((image.shape[1], image.shape[2]), device=device, dtype=torch.bool)
            for i in range(3):
                mono_albedo = vis[i]
                min_value, max_value = min_thrd, max_thrd
                filter &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            
            filtered = vis * (mask.expand_as(vis).type(mask.dtype))
            filtered_images.append(filtered)
            mean_value = torch.zeros(3)        
            for i in range(3):
                mono_albedo = vis[i]
                selected_values = mono_albedo[filter]
                if selected_values.numel() > 0:
                    mean_value[i] = selected_values.median()
                else:
                    mean_value[i] = float('nan')  # 값이 없을 경우 NaN
            
            flat_skin = torch.zeros_like(vis)
            mean_skin = torch.zeros_like(vis)
            for i in range(3):
                flat_skin[i, :, :] = mask * mean_value[i]

            mean_skin = (vis + flat_skin) / 2.0
            face_filter = mask.expand_as(image)
            enhanced = image * (1 - face_filter) + mean_skin * face_filter
            enhanced_images.append(enhanced)
            mean_values.append(mean_value)

        return {
            'segmented_masks': seg_probs,
            'segmented_images': vis_images,
            'enhanced_images': torch.stack(enhanced_images),
            'filtered_images': torch.stack(filtered_images),
            'mean_values': torch.stack(mean_values)
        }
    
    def paint_skin(self, rgbs):
        converted = []
        rgbs = rgbs.to(self.body_albedo.device)
        for rgb in rgbs:
            albedo = copy.deepcopy(self.body_albedo)
            diff = rgb - torch.tensor([205., 205., 205.]).to(rgb.device)
            painted = torch.clamp(albedo + diff.view(-1, 1, 1), 0, 255)
            converted.append(painted)
        return torch.stack(converted)
    
    def paint_with_mask(self, images: torch.Tensor, mask: torch.Tensor, colors: torch.Tensor):
        """_summary_

        Args:
            images (torch.Tensor): (b, c, w, h)
            masks (torch.Tensor): (b, w, h)
            color (torch.Tensor): (b, 3)
        """
        painted_images = []
        colors = colors.to(images.device)
        mask = mask.to(images.device)
        for image, color in zip(images, colors):
            partial = color.view(-1, 1, 1) * mask.expand_as(image)
            painted = image * (1-mask.expand_as(image)) + partial
            painted_images.append(painted)
        return torch.stack(painted_images)
    
    def decouple_head_albedo(self, albedo):
        device = albedo.device
        dtype = albedo.dtype

        head_albedo = torch.zeros((albedo.shape[0], 3, 256, 256), dtype=dtype, device=device, requires_grad=False)
        
        x_coords = torch.tensor(self.bridge['x_coords'], dtype=torch.int32, device=device, requires_grad=False)
        y_coords = torch.tensor(self.bridge['y_coords'], dtype=torch.int32, device=device, requires_grad=False)
        target_pixel_ids = torch.tensor(self.bridge['target_pixel_ids'], dtype=torch.int32, device=device, requires_grad=False)
        source_uv_points = torch.tensor(self.bridge['source_uv_points'], dtype=torch.float32, device=device, requires_grad=False)
        source_tex_coords = torch.zeros((source_uv_points.shape[0], source_uv_points.shape[1]), dtype=torch.int32, device=device, requires_grad=False)
        source_tex_coords[:, 0] = torch.clamp(head_albedo.shape[2]*(1.0-source_uv_points[:,1]), 0.0, head_albedo.shape[2]).type(torch.IntTensor)
        source_tex_coords[:, 1] = torch.clamp(head_albedo.shape[3]*(source_uv_points[:,0]), 0.0, head_albedo.shape[3]).type(torch.IntTensor)
        
        head_albedo[:, :, source_tex_coords[:,0], source_tex_coords[:,1]] = albedo[:, :, y_coords[target_pixel_ids], x_coords[target_pixel_ids]]
        return copy.deepcopy(head_albedo)
    
    def to_body_texture(self, partial_albedo: torch.Tensor):
        batch_size = partial_albedo.shape[0]
        channel = partial_albedo.shape[1]
        device = partial_albedo.device
        dtype = partial_albedo.dtype

        body_albedo = torch.zeros((batch_size, channel, 1024, 1024), dtype=dtype, device=device, requires_grad=False)
        x_coords = torch.tensor(self.bridge['x_coords'], dtype=torch.int32, device=device, requires_grad=False)
        y_coords = torch.tensor(self.bridge['y_coords'], dtype=torch.int32, device=device, requires_grad=False)
        target_pixel_ids = torch.tensor(self.bridge['target_pixel_ids'], dtype=torch.int32, device=device, requires_grad=False)
        source_uv_points = torch.tensor(self.bridge['source_uv_points'], dtype=torch.float32, device=device, requires_grad=False)
        source_tex_coords = torch.zeros((source_uv_points.shape[0], source_uv_points.shape[1]), dtype=torch.int32, device=device, requires_grad=False)
        source_tex_coords[:, 0] = torch.clamp(partial_albedo.shape[2]*(1.0-source_uv_points[:,1]), 0.0, partial_albedo.shape[2]).type(torch.IntTensor)
        source_tex_coords[:, 1] = torch.clamp(partial_albedo.shape[3]*(source_uv_points[:,0]), 0.0, partial_albedo.shape[3]).type(torch.IntTensor)

        body_albedo[:, :, y_coords[target_pixel_ids], x_coords[target_pixel_ids]] = partial_albedo[:, :, source_tex_coords[:,0], source_tex_coords[:,1]]
        return copy.deepcopy(body_albedo)
    
    def couple_body_texture(self, head_albedo: torch.Tensor, body_albedo: torch.Tensor):
        """_summary_
        Args:
            head_albedo (torch.Tensor): shape (c, 512, 512)
            body_albedo (torch.Tensor): shape (c, 1024, 1024)
        """
        device = head_albedo.device
        dtype = head_albedo.dtype

        x_coords = torch.tensor(self.bridge['x_coords'], dtype=torch.int32, device=device, requires_grad=False)
        y_coords = torch.tensor(self.bridge['y_coords'], dtype=torch.int32, device=device, requires_grad=False)
        target_pixel_ids = torch.tensor(self.bridge['target_pixel_ids'], dtype=torch.int32, device=device, requires_grad=False)
        source_uv_points = torch.tensor(self.bridge['source_uv_points'], dtype=torch.float32, device=device, requires_grad=False)
        source_tex_coords = torch.zeros((source_uv_points.shape[0], source_uv_points.shape[1]), dtype=torch.int32, device=device, requires_grad=False)
        source_tex_coords[:, 0] = torch.clamp(head_albedo.shape[2]*(1.0-source_uv_points[:,1]), 0.0, head_albedo.shape[2]).type(torch.IntTensor)
        source_tex_coords[:, 1] = torch.clamp(head_albedo.shape[3]*(source_uv_points[:,0]), 0.0, head_albedo.shape[3]).type(torch.IntTensor)

        body_albedo[:, :, y_coords[target_pixel_ids], x_coords[target_pixel_ids]] = head_albedo[:, :, source_tex_coords[:,0], source_tex_coords[:,1]]
        return body_albedo
    
class NakedHuman:
    def __init__(self, device="cuda:0"):
        import facer
        from creadto.models.recon import DetailFaceModel
        model_root = "creadto-model"
        self.flaep = DetailFaceModel()
        self.face_detector = facer.face_detector('retinaface/mobilenet',
                                                 device=device,
                                                 model_path=osp.join(model_root, "mobilenet0.25_Final.pth"))
        self.face_parser = facer.face_parser('farl/celebm/448',
                                             model_path=osp.join(model_root, "face_parsing.farl.celebm.main_ema_181500_jit.pt"),
                                             device=device) # optional "farl/lapa/448"
    
        self.bridge = np.load(osp.join(model_root, "flame", "flame2smplx_tex_1024.npy"), allow_pickle=True, encoding = 'latin1').item()
        self.head_map = torch.load(osp.join(model_root, "textures", "head_texture_map.pt"))
        self.body_map = torch.load(osp.join(model_root, "textures", "body_texture_map.pt"))
        self.mst = np.load(osp.join(model_root, "textures", "MSTScale", "MSTScaleRGB.npy"))
        self.mst = np.delete(self.mst, len(self.mst)-1, 0)
        self.mst = np.delete(self.mst, len(self.mst)-1, 0)
        # self.mst_bias = {'lower': 10, 'upper': 60}
        self.mst_bias = {'lower': 0, 'upper': 30}
    
        self.device = torch.device(device)
        
    def __call__(self, images: List[Image]):
        down_sample = transforms.Compose([transforms.Resize((256, 256))])
        up_sample = transforms.Compose([transforms.Resize((512, 512))])
        vis_dict = dict()
        crop_images, process = self.flaep.encode_pil(images)
        enhance_dict = self.enhance_skin(crop_images)
        result = self.flaep.decode(crop_images, external_img=enhance_dict["enhanced_images"] / 255.)
        
        # pre_texture, _ = self.make_head(result["uv_texture_gt"])
        # result = self.flaep.decode(crop_images, external_tex=down_sample(pre_texture))
        head_albedo, tone_indices = self.make_head(result["uv_texture_gt"], skin_values=enhance_dict["skin_values"] / 255.)
        full_texture = self.map_body(head_albedo, tone_indices, tone_values=enhance_dict["skin_values"] / 255.)
        
        vis_dict = {
            'head_images': crop_images,
            'face_detection': process,
            'head_texture': head_albedo,
            'full_texture': full_texture
        }
        vis_dict.update(enhance_dict)
        return vis_dict
    
    def enhance_skin(self, images):
        images = images * 255.
        device = images.device
        categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3,
                "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                "hat": 16, "earring": 17, "necklace": 18}
            
        with torch.inference_mode():
            faces = self.face_detector(images)
            faces = self.face_parser(images, faces)
        
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)
        vis_seg_probs = seg_probs.argmax(dim=1).float()/len(categories)*255
        vis_img = vis_seg_probs.sum(0, keepdim=True)
        
        face_masks = seg_probs[:, categories['skin']] + seg_probs[:, categories['nose']]
        
        enhanced_images = []
        skin_values = []
        detail_mask = []
        for image, face_mask in zip(images, face_masks):
            vis = image * face_mask.expand_as(image)
            
            mask = torch.ones((image.shape[1], image.shape[2]), device=device, dtype=torch.bool)
            for i in range(3):
                mono_albedo = vis[i]
                min_value, max_value = self.mst.min(axis=0)[i], self.mst.max(axis=0)[i] + self.mst_bias['upper']
                mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            
            mean_values = torch.zeros(3)        
            for i in range(3):
                mono_albedo = vis[i]
                selected_values = mono_albedo[mask]
                if selected_values.numel() > 0:
                    mean_values[i] = selected_values.median()
                else:
                    mean_values[i] = float('nan')  # 값이 없을 경우 NaN
            diff = torch.tensor(self.mst, dtype=mean_values.dtype, device=mean_values.device) - mean_values 
            print(diff)
            tone_index = torch.argmin(abs(diff).sum(dim=1))
            # mst_value = mst[tone_index]
            mst_value = mean_values
            print(mst_value)
            flat_skin = torch.zeros_like(vis)
            mean_skin = torch.zeros_like(vis)
            for i in range(3):
                flat_skin[i, :, :] = face_mask * mst_value[i]

            mean_skin = (vis + flat_skin) / 2.0
            face_filter = face_mask.expand_as(image)
            enhanced = image * (1 - face_filter) + mean_skin * face_filter
            enhanced_images.append(enhanced)
            skin_values.append(mst_value)
            detail_mask.append(mask)
        
        return {'enhanced_images': torch.stack(enhanced_images),
                'skin_values': torch.stack(skin_values),
                'face_masks': face_masks,
                'detail_masks': torch.stack(detail_mask)}
        
    def map_body(self, head_albedos: torch.Tensor, tone_indices: torch.Tensor, tone_values: torch.Tensor = None):
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
        pack = zip(head_albedos, mst[tone_indices])
        if tone_values is not None:
            tone_values = tone_values.to(device)
            pack = zip(head_albedos, tone_values)
        for head_albedo, tone in pack:
            body_albedo = copy.deepcopy(self.body_map['default_albedo'])
            base_mask = torch.ones((body_albedo.shape[1], body_albedo.shape[2]), device=mst.device, dtype=torch.bool, requires_grad=False)
            for i in range(3):
                mono_albedo = body_albedo[i]
                min_value, max_value = mst.min(dim=0)[0][i] - (self.mst_bias['lower'] / 255.), mst.max(dim=0)[0][i] + (self.mst_bias['upper'] / 255.)
                base_mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            
            base_basis = base_mask.unsqueeze(dim=0).expand_as(body_albedo) * (tone - self.body_map['mean']).view(-1, 1, 1)
            body_albedo = torch.clamp(body_albedo + base_basis, 0.0, 1.0)
            
            source_tex_coords = torch.zeros((source_uv_points.shape[0], source_uv_points.shape[1]), dtype=torch.int32, device=device, requires_grad=False)
            source_tex_coords[:, 0] = torch.clamp(head_albedo.shape[1]*(1.0-source_uv_points[:,1]), 0.0, head_albedo.shape[1]).type(torch.IntTensor)
            source_tex_coords[:, 1] = torch.clamp(head_albedo.shape[2]*(source_uv_points[:,0]), 0.0, head_albedo.shape[2]).type(torch.IntTensor)

            body_albedo[:, y_coords[target_pixel_ids], x_coords[target_pixel_ids]] = head_albedo[:, source_tex_coords[:,0], source_tex_coords[:,1]]
            body_albedos.append(body_albedo)
        
        return body_albedos
            
    def make_head(self, albedo: torch.Tensor, skin_values: torch.Tensor = None):
        """_summary_

        Args:
            albedo (torch.Tensor): (b, 3, 512, 512) albedo images(for head)
        """
        resize = transforms.Compose([transforms.Resize((512, 512))])
        albedo = resize(albedo)

        # skin_albedo = albedo * self.head_map['skin_mask'].to(self.device)
        skin_albedo = albedo
        self.save_images("skin-image.jpg", skin_albedo)
        tone_indices, tone_diff = self.detect_skin(skin_albedo=skin_albedo)
        fetched = self.fetch_skin(albedo=albedo, indices=tone_indices, tone_diffs=tone_diff, skin_values=skin_values)
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
            
    def fetch_skin(self, albedo: torch.Tensor, indices: torch.Tensor, tone_diffs: torch.Tensor, skin_values: torch.Tensor = None) -> torch.Tensor:
        for key in self.head_map:
            self.head_map[key] = self.head_map[key].to(albedo.device)
        mst = torch.tensor(self.mst, dtype=tone_diffs.dtype, device=albedo.device) / 255.
        fetched_result = []
        if skin_values is not None:
            skin_values = skin_values.to(albedo.device)
            pack = zip(albedo, skin_values, tone_diffs)
        else:
            pack = zip(albedo, mst[indices], tone_diffs)
        for skin, base_diff, diff in pack:
            face_mask = torch.ones((albedo.shape[2], albedo.shape[3]), device=mst.device, dtype=torch.bool)
            base_mask = torch.ones((self.head_map['default_albedo'].shape[1], self.head_map['default_albedo'].shape[2]), device=mst.device, dtype=torch.bool)
            for i in range(3):
                mono_albedo = skin[i]
                mono_base = self.head_map['default_albedo'][i]
                min_value, max_value = mst.min(dim=0)[0][i] - (self.mst_bias['lower'] / 255.), mst.max(dim=0)[0][i] + (self.mst_bias['upper'] / 255.)
                face_mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
                base_mask &= (mono_base >= min_value) & (mono_base <= max_value)
            
            basis_median = torch.zeros(3).to(albedo.device)
            for i in range(3):
                mono_albedo = skin[i]
                selected_values = mono_base[base_mask]
                if selected_values.numel() > 0:
                    basis_median[i] = selected_values.median()
                else:
                    basis_median[i] = float('nan')  # 값이 없을 경우 NaN
                    
            face_basis = face_mask.unsqueeze(dim=0).expand_as(skin) * diff.view(-1, 1, 1)
            base_basis = base_mask.unsqueeze(dim=0).expand_as(skin) * (base_diff - basis_median).view(-1, 1, 1)
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
                min_value, max_value = self.mst.min(axis=0)[i] - self.mst_bias['lower'], self.mst.max(axis=0)[i] + self.mst_bias['upper']
                min_value, max_value = min_value / 255., max_value / 255.
                mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            
            for i in range(3):
                mono_albedo = skin[i]
                selected_values = mono_albedo[mask]
                if selected_values.numel() > 0:
                    mean_values[i] = selected_values.median()
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


def make_head_map():
    import torchvision.io as io
    from torchvision.transforms import ToTensor
    import cv2
    
    model_root = "creadto-model"
    albedo_path = r"D:\Creadto\CreadtoLibrary\creadto-model\textures\MSTScale\Samples\default_head (1).png"
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
        'mean': torch.tensor([0.9519, 0.9268, 0.8955], dtype=torch.float32),
        'skin_mask': skin_mask.expand_as(default_albedo),
        'face_mask': face_mask.expand_as(default_albedo),
        'contour_mask': contour_mask.expand_as(default_albedo),
    }
    torch.save(head_texture_kit, "head_texture_map.pt")

def make_body_map():
    pass

if __name__ == "__main__":
    make_head_map()