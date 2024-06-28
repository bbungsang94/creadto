import platform
import os

if platform.system().lower() == "linux":
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    root = r"/workspace/cache/template/high-texture-raw"
    texture_root = "/workspace/cache/template/high-texture-raw/white"
else:
    root = r"/workspace/cache/template/high-texture-raw"
    texture_root = "/workspace/cache/template/high-texture-raw/white"
        
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
for face in tqdm(uv_faces):
    a = uv_coords[face[0]] * width
    b = uv_coords[face[1]] * width
    c = uv_coords[face[2]] * width
    
    backboard = cv2.line(backboard, (int(a[0]), int(a[-1])), (int(b[0]), int(b[-1])), color)
    backboard = cv2.line(backboard, (int(b[0]), int(b[-1])), (int(c[0]), int(c[-1])), color)
    backboard = cv2.line(backboard, (int(a[0]), int(a[-1])), (int(c[0]), int(c[-1])), color)

cv2.imwrite('backboard.png', backboard)


# Filtering Section
import torch
from creadto.utils.math import is_in_polygon
image_file = "temp_white_8k.png"

texture_image = cv2.imread(osp.join(texture_root, image_file), cv2.IMREAD_COLOR)
import glob
resume_files = glob.glob('./dump/*.png')
resume_count = -1
backboard = np.zeros_like(texture_image)
if len(resume_files) > 0:
    resume_files.sort(key=os.path.getmtime)
    iteration_name = resume_files[-1].split('-')[-1].split('.')[0]
    resume_count = int(iteration_name)

    backboard = cv2.imread(resume_files[-1], cv2.IMREAD_COLOR)
width = backboard.shape[0]
height = backboard.shape[1]

pbar = tqdm(uv_faces)
length = uv_faces.shape[0]
save_interval = 5000
for i, face in enumerate(pbar):
    pbar.set_description("%d %0.2f%% processed" % (i, (i / length) * 100))
    if resume_count >= i:
        continue
    a = uv_coords[face[0]]
    b = uv_coords[face[1]]
    c = uv_coords[face[2]]
    vertex = torch.stack([a, b, c])
    max_val, _ = vertex.max(dim=0)
    min_val, _ = vertex.min(dim=0)
    x_range = range(int(min_val[0] * width), int(max_val[0] * width) + 1)
    y_range = range(int(min_val[1] * height), int(max_val[1] * height) + 1)
    for x in x_range:
        for y in y_range:
            value = backboard[y, x, :]
            point = (x / width, y / height)
            if is_in_polygon(vertex, point):
                backboard[y, x, :] = texture_image[y, x, :]
    if i % save_interval == 0:
        cv2.imwrite('./dump/backboard-%d.png' % i, backboard)
cv2.imwrite('./dump/final_backboard.png', backboard)