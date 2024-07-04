import cv2
import numpy as np
from tqdm import tqdm

def compute_normal_map(image_path, scale=1.0, bias=0.5):
    # 이미지를 그레이스케일로 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 이미지의 크기 가져오기
    h, w = img.shape

    # Sobel 필터를 사용하여 이미지의 X 및 Y 방향 기울기 계산
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # 노멀 벡터 계산
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    
    # b g r to r g b
    normal_map[..., 2] = sobel_x
    normal_map[..., 1] = sobel_y
    normal_map[..., 0] = 1.0 / scale_factor
    
    # 노멀라이즈
    norm = np.sqrt(np.sum(normal_map ** 2, axis=2, keepdims=True))
    normal_map = normal_map / norm
    
    # 스케일 및 바이어스 적용
    normal_map = (normal_map * 0.5 + 0.5) * 255.0 * bias
    normal_map = np.clip(normal_map, 0, 255)
    
    return normal_map.astype(np.uint8)

# 이미지 경로 설정
image_path = r'D:\Creadto\CreadtoLibrary\creadto-model\template\high-texture-raw\white\white_m_8k_raw.png'
n_split = 10
min_scale_factor = 0.0
max_scale_factor = 2.0
for itr in tqdm(range(1, n_split + 1)):
    scale_factor = min_scale_factor + (itr / n_split) * max_scale_factor
    # normal map 생성
    normal_map = compute_normal_map(image_path, scale=scale_factor, bias=0.9)

    # 결과 저장
    cv2.imwrite('./dump/normal_map_%08d.png' % int(scale_factor * n_split), normal_map)