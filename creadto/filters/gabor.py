import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GaborFilter:
    def __init__(self, kernel_size=31, sigma=4.0, theta=np.pi/4, lambda_=10.0, gamma=0.5, psi=0):
        self.kernel = self._make_kernel(kernel_size, sigma, theta, lambda_, gamma, psi)
        self.kernel = self.kernel[None, None, :]  # Shape it to [out_channels, in_channels, height, width]
    
    def __call__(self, x):
        # shape of image x is [batch_size, channel, width, height]
        filtered_images = []
        for idx in range(x.shape[1]):
            gray = x[:, idx, :]
            filtered_images.append(F.conv2d(gray[:, None, :], self.kernel, padding=self.kernel.shape[0]//2, stride=1))
            
        return torch.cat(filtered_images, dim=1)
    
    def _make_kernel(self, kernel_size, sigma, theta, lambda_, gamma, psi):
        theta = torch.tensor(theta)
        y, x = torch.meshgrid([
            torch.arange(-(kernel_size // 2), kernel_size // 2 + 1),
            torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
            ])
    
        rot_x = x * torch.cos(theta) + y * torch.sin(theta)
        rot_y = -x * torch.sin(theta) + y * torch.cos(theta)
        
        gabor_kernel = torch.exp(-0.5 * (rot_x ** 2 + gamma ** 2 * rot_y ** 2) / sigma ** 2)
        gabor_kernel *= torch.cos(2 * np.pi * rot_x / lambda_ + psi)
        gabor_kernel /= torch.sum(gabor_kernel)  # Normalize the kernel
        
        return gabor_kernel

if __name__ == "__main__":
    import os
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    # Gabor filter parameters
    param = {
        "kernel_size": 31 ,      # Kernel size
        "sigma": 4.0,      # Standard deviation of Gaussian
        "theta": np.pi / 4,  # Orientation
        "lambda_": 10.0,     # Wavelength of the sinusoidal factor
        "gamma": 0.5,      # Spatial aspect ratio
        "psi": 0,          # Phase offset
    }

    filter = GaborFilter(**param)
    in_trans = ToTensor()
    out_trans = ToPILImage()
    dummy_in = in_trans(Image.open(os.path.join(r"D:\dump\archive\cache\input-images", "male_western_0003.jpeg")))
    dummy_out = []
    for gray in dummy_in:
        dummy_out.append(filter(gray.unsqueeze(dim=0).unsqueeze(dim=0)))
    dummy_out = out_trans(torch.cat(dummy_out, dim=1)[0])
    dummy_out.save("Gabor-out-separate.jpg")
    dummy_out = out_trans(filter(dummy_in.unsqueeze(dim=0))[0])
    dummy_out.save("Gabor-out-whole.jpg")
    