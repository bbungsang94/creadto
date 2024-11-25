import os
import copy
import os.path as osp
import numpy as np
from PIL import Image
import torch
import facer
from torchvision.transforms.functional import to_tensor, resize, to_pil_image


def main():
    file_roots = {
        "image": r"D:\dump\eyes_color\input_images",
        "texture": r"D:\dump\eyes_color\input_texture",
        "segmented": r"D:\dump\eyes_color\debug_segmented",
        "output": r"D:\dump\eyes_color\output_texture",
    }
    parser_dict = get_face_parser()
    images, files = load_images(root=file_roots['image'])
    texture, mask = load_texture(root=file_roots['texture'])
    binary_mask = copy.deepcopy(mask)
    binary_mask[mask > 0] = 1.
    
    probs = parse_face(images.cuda(), **parser_dict)
    probs = probs.cpu()
    images = images.cpu()
    
    masks = torch.zeros((probs[:, 0].shape), device=probs.device, dtype=probs.dtype)
    eye_masks = masks + probs[:, parser_dict['categories']['left_eye']]
    eye_masks = eye_masks + probs[:, parser_dict['categories']['right_eye']]
    
    for i, pack in enumerate(zip(images, eye_masks)):
        image, eye_mask = pack
        eye_vis = image * eye_mask.expand_as(image)

        pil_image = to_pil_image(eye_vis)
        pil_image.save(osp.join(file_roots["segmented"], files[i]))

        image_np = eye_vis.permute(1, 2, 0).numpy()
        pixel_mask = np.any(image_np > 0, axis=-1)
        eye_pixels = image_np[pixel_mask]
        average_color = eye_pixels.mean(axis=0)
        
        temp_texture, temp_mask = copy.deepcopy(texture), copy.deepcopy(mask)
        iris_texture = torch.stack([temp_mask[0] * average_color[0], temp_mask[1] * average_color[1], temp_mask[2] * average_color[2]])
        save_texture = temp_texture[:3] * (1 - binary_mask[:3]) + iris_texture * binary_mask[:3]
        
        pil_image = to_pil_image(save_texture)
        pil_image.save(osp.join(file_roots["output"], files[i]))
        
def parse_face(images, detector, parser, **kwargs):
    # bs, 3, 224, 244 in [0, 255]
    images = images * 255
    with torch.inference_mode():
        faces = detector(images)
        faces = parser(images, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    
    return torch.clamp(seg_probs * 2 - 1, 0., 1.)

def get_face_parser(model_root="D:\Creadto\CreadtoLibrary\creadto-model", device="cuda:0"):
    face_detector = facer.face_detector('retinaface/mobilenet', device=device,
                                            model_path=osp.join(model_root, "mobilenet0.25_Final.pth"))
    face_parser = facer.face_parser('farl/celebm/448', device=device,
                                            model_path=osp.join(model_root, "face_parsing.farl.celebm.main_ema_181500_jit.pt")) # optional "farl/lapa/448"
    categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3, 
                  "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                  "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                  "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                  "hat": 16, "earring": 17, "necklace": 18}
    return {'detector': face_detector, 'parser': face_parser, 'categories': categories}

def load_images(root=r"D:\dump\eyes_color\input_images"):
    files = os.listdir(root)
    images = []
    for file in files:
        image = Image.open(osp.join(root, file))
        images.append(to_tensor(image))
    return torch.stack(images, dim=0), files   

def load_texture(root=r"D:\dump\eyes_color\input_texture"):
    texture = to_tensor(Image.open(osp.join(root, "eye-texture.png")))
    mask = to_tensor(Image.open(osp.join(root, "eye-mask.png")))
    return texture, mask

if __name__ == "__main__":
    main()