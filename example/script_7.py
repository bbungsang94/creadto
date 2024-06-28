import os.path as osp

root = r"/workspace/cache/template/high-texture-raw"
filename = "high-resolution-human.obj"

from creadto.utils.io import load_mesh

v, f, uv_coords, uv_faces = load_mesh(osp.join(root, filename))

import numpy as np
backboard = np.ones((8192, 8192, 3)) * 255.

import cv2
width = backboard.shape[0]
height = backboard.shape[1]
color = (10, 10, 10)

uv_coords[:, -1] = 1. - uv_coords[:, -1]
from tqdm import tqdm
# for face in tqdm(uv_faces):
#     a = uv_coords[face[0]] * width
#     b = uv_coords[face[1]] * width
#     c = uv_coords[face[2]] * width
    
#     backboard = cv2.line(backboard, (int(a[0]), int(a[-1])), (int(b[0]), int(b[-1])), color)
#     backboard = cv2.line(backboard, (int(b[0]), int(b[-1])), (int(c[0]), int(c[-1])), color)
#     backboard = cv2.line(backboard, (int(a[0]), int(a[-1])), (int(c[0]), int(c[-1])), color)

cv2.imwrite('backboard.png', backboard)


# Filtering Section
import torch
from creadto.utils.math import is_in_polygon
texture_root = "workspace/cache/template/high-texture-raw/white"
image_file = "temp_white_8k.png"

texture_image = cv2.imread(osp.join(texture_root, image_file), cv2.IMREAD_COLOR)
backboard = np.zeros_like(texture_image)
width = backboard.shape[0]
height = backboard.shape[1]

pbar = tqdm(range(width))
length = uv_faces.shape[0]
for w in pbar:
    for h in range(height):
        point = (w / width, h / height)
        for i, face in enumerate(uv_faces):
            pbar.set_description("%d, %d %0.2f%% processed" % (w, h, (i / length) * 100))
            a = uv_coords[face[0]]
            b = uv_coords[face[1]]
            c = uv_coords[face[2]]
            
            if is_in_polygon(torch.stack([a, b, c]), point):
                backboard[w, h, :] = texture_image[w, h, :]
    cv2.imwrite('./dump/backboard-%d.png' % w, backboard)
cv2.imwrite('backboard.png', backboard)