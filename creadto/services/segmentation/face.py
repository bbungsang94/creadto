from typing import Dict, List
import facer
import torch
from creadto.services.detection.face import FacialDetector
from creadto.utils.decorator import get_instances, service
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images


@service
class FacialSegmenter:
    def __init__(self):
        device = devicex()
        self.categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3,
                           "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                           "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                           "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                           "hat": 16, "earring": 17, "necklace": 18}
        
        self.face_detector = facer.face_detector('retinaface/mobilenet',
                                                 device=device,
                                                 model_path=r"./creadto-model/detection/facer/mobilenet0.25_Final.pth")
        self.face_parser = facer.face_parser('farl/celebm/448',
                                             device=device,
                                             model_path=r"./creadto-model/segmentation/facer/face_parsing.farl.celebm.main_ema_181500_jit.pt") # optional "farl/lapa/448"
        
        self.detector = FacialDetector(device=device)
    
    def __call__(self, images):
        head_images = self.detector(images)
        with torch.inference_mode():
            faces = self.face_detector(head_images * 255.)
            faces = self.face_parser(head_images, faces)
        
        drawn_image = facer.draw_bchw(head_images, faces)
        faces['seg'].update({'vis_image': drawn_image, 'head_image': head_images})
        return faces['seg']
    
    def get_parts_colour(self, segments: Dict[str, torch.Tensor], parts: List[int], min_thrd = 5, max_thrd = 170) -> Dict[str, object]:
        """From segment results, extract parts colour.

        Args:
            segments dict(str, tensor): logits of segmentation
            parts (List[int]): list of parts to extract colour
            min_thrd (int, optional): for mask of sclera. Defaults to 5.
            max_thrd (int, optional): _for mask of sclera. Defaults to 170.

        Returns:
            Dict[str, object]: Dict of operation results such as input images, processed images, colour values, and file names
        """
        
        device = segments['logits'].device
        seg_logits = segments['logits']
        seg_probs = seg_logits.softmax(dim=1)
        
        masks = torch.zeros((seg_probs[:, 0].shape), device=device, dtype=seg_probs.dtype)
        for part in parts:
            masks += seg_probs[:, part]
        mean_values = []
        enhanced_images = []
        for image, mask in zip(segments['head_image'], masks):
            vis = image * mask.expand_as(image)
            mean_value = torch.zeros(3, device=device)
            flat_skin = torch.zeros_like(vis)
            
            filter = torch.ones((image.shape[1], image.shape[2]), device=device, dtype=torch.bool)
            for i in range(3):
                mono_albedo = vis[i]
                min_value, max_value = min_thrd, max_thrd
                filter &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
                selected_values = mono_albedo[filter]
                if selected_values.numel() > 0:
                    mean_value[i] = selected_values.median()
                else:
                    mean_value[i] = float('nan')  # 값이 없을 경우 NaN
                flat_skin[i, :, :] = mask * mean_value[i]
            
            mean_skin = (vis + flat_skin) / 2.0
            face_filter = mask.expand_as(image)
            enhanced = image * (1 - face_filter) + mean_skin * face_filter
            enhanced_images.append(enhanced)
            mean_values.append(mean_value)

        return {
            'segmented_masks': seg_probs,
            'enhanced_images': torch.stack(enhanced_images, device=device),
            'mean_values': torch.stack(mean_values, device=device)
        }
            
def _get_current_(cls):
    instance = get_instances(cls)
    if instance == None:
        instance = cls()
    return instance

def segment_face_from_directory(directory: str) -> Dict[str, object]:
    images = load_images(directory, integer=True, pad=True)
    images = images.permute(0, 2, 3, 1)
    return segment_from_tensor(images)
    
def segment_face_from_file(image_path: str) -> Dict[str, object]:
    image = load_image(image_path, integer=True).squeeze()
    image = image.permute(1, 2, 0)
    
    return segment_from_tensor(image.unsqueeze(dim=0))

def segment_from_tensor(image: torch.Tensor) -> Dict[str, object]:
    image = image.to(devicex())
    return _get_current_(FacialSegmenter)(image)