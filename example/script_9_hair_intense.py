import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
import facer
from torchvision.transforms.functional import to_tensor, resize, to_pil_image
from creadto.filters.gabor import GaborFilter

def main():
    output_root = r"D:\dump\cache\processed-images"
    param = {
        "kernel_size": 7 ,      # Kernel size
        "sigma": 1.5,      # Standard deviation of Gaussian
        "theta": np.pi / 4,  # Orientation
        "lambda_": 10.0,     # Wavelength of the sinusoidal factor
        "gamma": 0.5,      # Spatial aspect ratio
        "psi": 0,          # Phase offset
    }
    filter = GaborFilter(**param)
    parser_dict = get_face_parser()
    images, files = load_images()
    images = resize(images, (224, 224)).to("cuda:0")
    
    probs = parse_face(images, **parser_dict)
    probs = probs.cpu()
    images = images.cpu()
    
    masks = torch.zeros((probs[:, 0].shape), device=probs.device, dtype=probs.dtype)
    hair_masks = masks + probs[:, parser_dict['categories']['hair']]
    face_masks = masks + probs[:, parser_dict['categories']['skin']]
    
    for i, pack in enumerate(zip(images, hair_masks, face_masks)):
        image, hair_mask, face_mask = pack
        hair_vis = image * hair_mask.expand_as(image)
        red_filtered = filter(hair_vis[0][None, None, :])
        red_filtered = resize(red_filtered, face_mask.shape)
        
        processed = red_filtered.squeeze()
        processed = torch.stack([processed / 1.5, processed / 2.0, processed])
        face_mask = torch.stack([face_mask / 2, face_mask / 3, face_mask / 4])
        processed = gasuss_noise(processed + face_mask)
        pil_image = to_pil_image(processed)
        pil_image.save(osp.join(output_root, files[i].replace("png", "jpg")))


def load_images(root=r"D:\dump\cache\hair-input-images"):
    files = os.listdir(root)
    images = []
    for file in files:
        image = Image.open(osp.join(root, file))
        images.append(to_tensor(image))
    return torch.stack(images, dim=0), files    
    
def get_face_parser(model_root="D:\Creadto\CreadtoLibrary\creadto-model", device="cuda:0"):
    face_detector = facer.face_detector('retinaface/mobilenet', device=device,
                                            model_path=osp.join(model_root, "detection/facer/mobilenet0.25_Final.pth"))
    face_parser = facer.face_parser('farl/celebm/448', device=device,
                                            model_path=osp.join(model_root, "segmentation/facer/face_parsing.farl.celebm.main_ema_181500_jit.pt")) # optional "farl/lapa/448"
    categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3, 
                  "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                  "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                  "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                  "hat": 16, "earring": 17, "necklace": 18}
    return {'detector': face_detector, 'parser': face_parser, 'categories': categories}

def parse_face(images, detector, parser, **kwargs):
    # bs, 3, 224, 244 in [0, 255]
    images = images * 255
    with torch.inference_mode():
        faces = detector(images)
        faces = parser(images, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    
    return torch.clamp(seg_probs * 2 - 1, 0., 1.)

def gasuss_noise(images, mean=0, var=0.001):
    noise = torch.normal(mean, var ** 0.5, size=images.shape)
    noisy_img = images + noise
    
    return noisy_img

if __name__ == "__main__":
    main()