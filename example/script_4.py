import copy
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import transforms, ToPILImage

from creadto._external.deca.decalib import DECA
from creadto._external.deca.decalib import FLAME
from creadto._external.deca.decalib import cfg as deca_cfg
from creadto._external.deca.decalib import SRenderY

from PIL import Image
from creadto.utils.math import is_in_polygon


def note():
    """
    이제서야 올바른 래스터화를 알게되어 본격적인 포토그래메트리 방식 텍스쳐 작업 스크립트
    :return:
    """
    pass


def main(root=r"D:\Creadto\creadto\dump\bm",
         mask_root=r"D:\Creadto\creadto\external\creadto\data\texture\mask(head)",
         flame_root=r"D:\Creadto\creadto\external\flame",
         mask_file="FLAME_masks.pkl",
         direction="right"):
    image_size = 256
    files = {"back": "back.jpg", "front": "front.jpg", "right": "right.jpg", "left": "left.jpg"}
    masks = {"front": "face_eyeball.jpg", "right": "right_ear.jpg", "left": "left_ear.jpg", "back": "scalp.jpg"}
    to_tensor = transforms.ToTensor()
    to_resized = transforms.Resize((image_size, image_size), antialias=True)
    to_pil = ToPILImage()
    flame_mask = np.load(os.path.join(flame_root, mask_file), allow_pickle=True, encoding="latin1")
    flame = FLAME(deca_cfg.model)
    deca = DECA(config=deca_cfg, device=torch.device("cpu"))
    render = SRenderY(image_size, obj_filename=deca_cfg.model.topology_path, uv_size=deca_cfg.model.uv_size,
                      rasterizer_type=deca_cfg.rasterizer_type)
    uvcoords = render.raw_uvcoords[0].cpu().numpy()
    faces = flame.faces_tensor
    uvfaces = render.uvfaces[0].cpu().numpy()
    encode_dict = proc_vertex(os.path.join(root, files['front']), deca)
    verts, landmarks2d, landmarks3d = flame(shape_params=encode_dict['shape'],
                                            expression_params=encode_dict['exp'],
                                            pose_params=encode_dict['pose'])
    opdict, visdict = deca.decode(encode_dict)
    uv_texture = opdict['uv_texture_gt']
    for direction in files.keys():
        if direction == "front":
            continue
        mask = to_tensor(Image.open(os.path.join(mask_root, masks[direction]))).unsqueeze(dim=0)
        pvert = get_pverts(flame_mask[masks[direction].replace('.jpg', '')], verts, faces, uvcoords, uvfaces)
        uv_gt = apply_pverts(root=root, image_name=files[direction], pvert=pvert)

        uv_texture = uv_gt[:, :3, :, :] * mask + (uv_texture[:, :3, :, :] * (1 - mask))
        pil_img = to_pil(uv_gt[0])
        pil_img.save(direction + "-uv_gt.jpg")
        pil_img = to_pil(uv_texture[0])
        pil_img.save(direction + "-uv_texture.jpg")

    opdict['uv_texture_gt'] = uv_texture
    deca.save_obj(os.path.join('result' + '.obj'), opdict)


def proc_vertex(path: str, deca):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    ])
    image = trans(Image.open(path)).unsqueeze(dim=0)
    with torch.no_grad():
        codedict = deca.encode(image)
    return codedict


def apply_pverts(root, image_name, pvert):
    to_tensor = transforms.ToTensor()
    to_resized = transforms.Resize((256, 256), antialias=True)
    to_pil = ToPILImage()
    image = to_resized(to_tensor(Image.open(os.path.join(root, image_name))).unsqueeze(dim=0))
    uv_gt = F.grid_sample(image, pvert[:, :, :, :2], mode='bilinear', align_corners=False)
    return uv_gt


def get_pverts(mask_index, vertex, faces, uvcoords, uvfaces):
    batch_size = 1
    image_size = 256
    to_pil = ToPILImage()
    partial_vertex = vertex[:, mask_index]
    max_value, _ = partial_vertex.max(dim=1)
    min_value, _ = partial_vertex.min(dim=1)
    vertex = (vertex - min_value) / (max_value - min_value)
    vertex = vertex * 2 - 1
    vertex[:, :, 1] = -1 * vertex[:, :, 1]

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(partial_vertex[0].cpu().detach().numpy())
    # o3d.visualization.draw_geometries([pcd])

    mapper_set = set()
    for i in mask_index:
        indexes, _ = np.where(faces == i)
        mapper_set = mapper_set.union(set(indexes))

    partial_uv_faces = uvfaces[list(mapper_set)]
    partial_faces = faces[list(mapper_set)]
    backboard = torch.zeros(batch_size, image_size, image_size, 2)
    for i in range(partial_uv_faces.shape[0]):
        uv_set = partial_uv_faces[i]
        xy_set = vertex[:, partial_faces[i]][:, :, :2]
        uv_region = copy.deepcopy(uvcoords[uv_set] * image_size)
        uv_min = uv_region.min(axis=0).astype(int)
        uv_max = uv_region.max(axis=0).astype(int)
        xy_min, _ = xy_set.min(dim=1)
        xy_max, _ = xy_set.max(dim=1)
        xy_gap = xy_max - xy_min
        for b in range(batch_size):
            for img_y in range(uv_min[1], uv_max[1] + 1):
                if uv_max[1] != uv_min[1]:
                    ratio_y = (img_y - uv_min[1]) / (uv_max[1] - uv_min[1])
                else:
                    ratio_y = 1.0
                for img_x in range(uv_min[0], uv_max[0] + 1):
                    if uv_max[0] != uv_min[0]:
                        ratio_x = (img_x - uv_min[0]) / (uv_max[0] - uv_min[0])
                    else:
                        ratio_x = 1.0
                    result = is_in_polygon(torch.tensor(uv_region), (img_x, img_y))
                    if result:
                        backboard[b, image_size - img_y, img_x, 1] = xy_min[b, 1] + (xy_gap[b, 1] * ratio_y)
                        backboard[b, image_size - img_y, img_x, 0] = xy_min[b, 0] + (xy_gap[b, 0] * ratio_x)

    x_uv = backboard[0, :, :, 0]
    x_uv = (x_uv - x_uv.min()) / (x_uv.max() - x_uv.min())
    pil_img = to_pil(x_uv)
    pil_img.save("temp-pverts-y.jpg")
    y_uv = backboard[0, :, :, 1]
    y_uv = (y_uv - y_uv.min()) / (y_uv.max() - y_uv.min())
    pil_img = to_pil(y_uv)
    pil_img.save("temp-pverts-x.jpg")
    return backboard


if __name__ == "__main__":
    main()
