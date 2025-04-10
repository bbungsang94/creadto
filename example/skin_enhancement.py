import os.path as osp
import numpy as np
import torch
from creadto.services.segmentation.face import FacialSegmenter
from creadto.utils.dist import devicex
from creadto.utils.io import load_image, load_images

def compute_mask(vis, mst, mst_bias, device):
    mask = torch.ones((vis.shape[1], vis.shape[2]), device=device, dtype=torch.bool)
    for i in range(3):
        min_value, max_value = mst.min(axis=0)[i], mst.max(axis=0)[i] + mst_bias['upper']
        mask &= (vis[i] >= min_value) & (vis[i] <= max_value)
    return mask

def compute_mean_values(vis, mask):
    mean_values = torch.zeros(3, device=vis.device, dtype=vis.dtype)
    for i in range(3):
        selected_values = vis[i][mask]
        mean_values[i] = selected_values.median() if selected_values.numel() > 0 else float('nan')
    return mean_values

def compute_tone_index(mean_values, mst):
    diff = torch.tensor(mst, dtype=mean_values.dtype, device=mean_values.device) - mean_values
    return torch.argmin(abs(diff).sum(dim=1))

def apply_tone_correction(image, face_mask, mean_values):
    flat_skin = face_mask * mean_values.view(3, 1, 1)
    mean_skin = (image * face_mask.expand_as(image) + flat_skin) / 2.0
    enhanced = image * (1 - face_mask.expand_as(image)) + mean_skin * face_mask.expand_as(image)
    return enhanced

def process_images(images, face_masks, mst, mst_bias, device):
    results = []
    tone_indices = []
    for image, face_mask in zip(images, face_masks):
        vis = image * face_mask.expand_as(image)
        mask = compute_mask(vis, mst, mst_bias, device)
        mean_values = compute_mean_values(vis, mask)
        tone_index = compute_tone_index(mean_values, mst)
        enhanced = apply_tone_correction(image, face_mask, mean_values)
        results.append(enhanced)
        tone_indices.append(tone_index)
    return results, tone_indices

def main(model_root = "./creadto-model"):
    from tqdm import tqdm
    import os
    import shutil
    import json
    from torchvision.transforms.functional import to_pil_image
    #directory = "./sample"
    device = devicex()
    segmenter = FacialSegmenter()
    
    root = r"D:\Creadto\Heritage\Dataset\CelebA\archive\img_align_celeba"
    scale_root = r"D:\Creadto\CreadtoLibrary\creadto-model\textures\MSTScale\MST Orbs"
    scale_path = os.listdir(scale_root)
    image_files = os.listdir(os.path.join(root, 'images'))
    for image_file in tqdm(image_files):
        label = {
            'image': 'images/' + image_file,
            'detection': 'detection/' + image_file,
            'segmentation': 'segmentation/' + image_file,
            'scale': 'scale/' + image_file,
            'label': -1,
        }
        images = load_image(os.path.join(root, 'images', image_file), integer=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to(device)

        if os.path.exists(os.path.join(root, 'labels', image_file.split('.')[0] + ".json")):
            continue
        
        try:
            result = segmenter(images)
            pil_image = to_pil_image(result['vis_image'][0].cpu())
            pil_image.save(os.path.join(root, label['segmentation']))
            result['head_image'] = result['head_image'] * 255.
            pil_image = to_pil_image(result['head_image'][0].cpu())
            pil_image.save(os.path.join(root, label['detection']))
            seg_logits = result['logits']
            seg_probs = seg_logits.softmax(dim=1)
            face_masks = seg_probs[:, segmenter.categories['skin']] + seg_probs[:, segmenter.categories['nose']]
            mst = np.load(osp.join(model_root, "textures", "MSTScale", "MSTScaleRGB.npy"))

            result, tone_indices = process_images(result['head_image'] * 255., face_masks, mst, {'lower': 0, 'upper': 30}, device)
            label['label'] = tone_indices[-1].item()
            shutil.copy(os.path.join(scale_root, scale_path[label['label']]), os.path.join(root, label['scale']))
            split_names = image_file.split('.')
            with open(os.path.join(root, 'labels', split_names[0] + ".json"), 'w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue


if __name__ == "__main__":
    main()