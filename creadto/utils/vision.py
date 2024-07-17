import torch
import numpy as np
from PIL.Image import Image
import cv2


def to_cv2(images: torch.Tensor | np.ndarray | Image, inverse_channel=True):
    if isinstance(images, torch.Tensor):
        images = images.cpu().detach().numpy()
    
    images = np.array(images)
    
    if images.max() < 1.1:
        images = images * 255.
    
    permute = (0, 2, 3, 1)
    _, height, _, channels = images.shape
            
    if channels > height:
        channels = height
        images = np.transpose(images, permute)
    
    if inverse_channel:
        # It's a samething both BGR2RGB to RGB2BGR
        if channels == 3:
            permute = [2, 1, 0]
            images = images[:, :, :, permute]
        elif channels == 4:
            permute = [2, 1, 0, 3]
            images = images[:, :, :, permute]
    
    return images

def to_tensor(images: np.ndarray):
    permute = [2, 1, 0]
    images = images[:, :, :, permute]
    
    permute = (0, 3, 1, 2)
    images = np.transpose(images, permute)
        
    images = torch.tensor(images, dtype=torch.float, requires_grad=False)
    images = images / 255.

    return images

def remove_light(image_bgr: np.ndarray, blur: int=75):
    # 1) RGB to LAB
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # 2) LAB with median 25 to 100 and bilateral
    image_mf = cv2.medianBlur(l_channel, blur)
    inverted_image = cv2.bitwise_not(image_mf)

    # add-weighted가 아닌 add를 하면 제일 높은 블럭을 찾을 수 있다.
    image_composite = cv2.addWeighted(l_channel, 0.5, inverted_image, 0.5, 0)

    # Light removal 완료
    remove_lab = cv2.merge([image_composite, a_channel, b_channel])
    remove_bgr = cv2.cvtColor(remove_lab, cv2.COLOR_LAB2BGR)
    return remove_bgr


def compute_normal_map(images: np.ndarray, scale: float=1.0, bias: float=0.5):
    # Get a image shape
    b, h, w, c = images.shape
    normal_maps = np.zeros_like(images)
    for i, image_bgr in enumerate(images):
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Calculate gradients oriented on X and Y using sobel filter
        sobel_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)

        # Set cacluated normal vector to map structure
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        normal_map[..., 2] = sobel_x
        normal_map[..., 1] = sobel_y
        normal_map[..., 0] = 1.0 / scale
        
        # Normalize
        norm = np.sqrt(np.sum(normal_map ** 2, axis=2, keepdims=True))
        normal_map = normal_map / norm
        
        # Apply scale and bias factors
        normal_map = (normal_map * 0.5 + 0.5) * 255.0 * bias
        normal_map = np.clip(normal_map, 0, 255)
        normal_maps[i] = normal_map
    
    return normal_maps.astype(np.uint8)