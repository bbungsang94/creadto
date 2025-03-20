import torch
import torch.nn.functional as F
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

def multi_band_blending(img1, img2, mask, levels=6):    
    class GaussianPyramid:
        def __init__(self, levels):
            self.levels = levels
            self.gaussian_kernel = self.create_gaussian_kernel()
        
        def create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
            # 1D Gaussian
            x = torch.arange(kernel_size).float() - kernel_size // 2
            gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
            gauss_1d /= gauss_1d.sum()
            # 2D Gaussian
            gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
            gauss_2d = gauss_2d.expand(3, 1, kernel_size, kernel_size)
            return gauss_2d

        def build_pyramid(self, image):
            pyramid = [image]
            current = image
            for _ in range(self.levels):
                blurred = F.conv2d(current, self.gaussian_kernel, padding=2, groups=3)
                down = F.avg_pool2d(blurred, kernel_size=2, stride=2)
                pyramid.append(down)
                current = down
            return pyramid

    class LaplacianPyramid:
        def __init__(self, levels):
            self.levels = levels
            self.gaussian_pyramid = GaussianPyramid(levels)
            self.gaussian_kernel = self.gaussian_pyramid.gaussian_kernel
        
        def build_pyramid(self, image):
            gp = self.gaussian_pyramid.build_pyramid(image)
            lp = []
            for i in range(self.levels):
                current = gp[i]
                expanded = F.interpolate(gp[i + 1], scale_factor=2, mode='bilinear', align_corners=False)
                # Ensure the expanded size matches the 'current' level
                if expanded.shape[-2:] != current.shape[-2:]:
                    expanded = F.pad(expanded, (0, current.shape[-1] - expanded.shape[-1], 0, current.shape[-2] - expanded.shape[-2]))
                laplacian = current - expanded
                lp.append(laplacian)
            lp.append(gp[-1])
            return lp

    def reconstruct_from_pyramid(lp_pyramid):
        image = lp_pyramid[-1]
        for i in range(len(lp_pyramid) - 2, -1, -1):
            image = F.interpolate(image, scale_factor=2, mode='bilinear', align_corners=False)
            # Ensure the expanded size matches the current 'Laplacian' level
            if image.shape[-2:] != lp_pyramid[i].shape[-2:]:
                image = F.pad(image, (0, lp_pyramid[i].shape[-1] - image.shape[-1], 0, lp_pyramid[i].shape[-2] - image.shape[-2]))
            image = image + lp_pyramid[i]
            image = torch.clamp(image, 0, 1)
        return image

    device = img1.device
    lp1 = LaplacianPyramid(levels).build_pyramid(img1)
    lp2 = LaplacianPyramid(levels).build_pyramid(img2)
    gp_mask = GaussianPyramid(levels).build_pyramid(mask)

    blended_pyramid = []
    for l1, l2, gm in zip(lp1, lp2, gp_mask):
        blended = l1 * gm + l2 * (1 - gm)
        blended_pyramid.append(blended)
    
    blended_image = reconstruct_from_pyramid(blended_pyramid)
    return blended_image