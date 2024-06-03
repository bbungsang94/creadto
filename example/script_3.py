import copy
import os
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torchvision.transforms import transforms, ToPILImage

from creadto._external.deca.decalib.deca import DECA
from creadto._external.deca.decalib.deca import FLAME, FLAMETex
from creadto._external.deca.decalib.utils import util
from creadto._external.deca.decalib.deca import cfg as deca_cfg
from creadto._external.deca.decalib.utils.renderer import SRenderY, set_rasterizer

from PIL import Image
import cv2


def mask2uv():
    image_size = 256
    device = torch.device("cuda")
    flame_root = r"D:\Creadto\creadto\external\flame"
    mask_file = "FLAME_masks.pkl"

    flame_mask = np.load(os.path.join(flame_root, mask_file), allow_pickle=True, encoding="latin1")
    flame = FLAME(deca_cfg.model).to(device)
    flametex = FLAMETex(deca_cfg.model).to(device)
    shape, exp, pose = torch.zeros(1, deca_cfg.model.n_shape), torch.zeros(1, deca_cfg.model.n_exp), torch.zeros(1,
                                                                                                                 deca_cfg.model.n_pose)
    verts, landmarks2d, landmarks3d = flame(shape_params=shape.to(device), expression_params=exp.to(device),
                                            pose_params=pose.to(device))
    faces = flame.faces_tensor.cpu().detach().numpy()

    set_rasterizer(deca_cfg.rasterizer_type)
    render = SRenderY(image_size, obj_filename=deca_cfg.model.topology_path, uv_size=deca_cfg.model.uv_size,
                      rasterizer_type=deca_cfg.rasterizer_type).to(device)

    uvcoords = render.raw_uvcoords[0].cpu().numpy()
    uvfaces = render.uvfaces[0].cpu().numpy()

    # test map
    for key, value in flame_mask.items():
        mapper_set = set()
        for v in value:
            indexes, _ = np.where(faces == v)
            mapper_set = mapper_set.union(set(indexes))

        backboard = np.zeros((image_size, image_size, 3))
        for index in mapper_set:
            for pivot in uvfaces[index]:
                uv = uvcoords[pivot]
                u, v = uv[0] * image_size, image_size - (uv[1] * image_size)
                backboard = cv2.circle(backboard, (int(u), int(v)), 1, (255, 255, 255), -1)
        cv2.imwrite(key + '.jpg', backboard)

    backboard = np.zeros((image_size, image_size, 3))
    for uv in uvcoords:
        u, v = uv[0] * image_size, image_size - (uv[1] * image_size)
        backboard = cv2.circle(backboard, (int(u), int(v)), 1, (255, 255, 255), -1)
    cv2.imwrite('total_face.jpg', backboard)


def make_mask_image(image_size=512):
    from creadto.utils.math import is_in_polygon
    device = torch.device("cuda")
    flame_root = r"./creadto-model"
    mask_file = "FLAME_masks.pkl"

    flame_mask = np.load("./creadto-model/flame/flame_masks.pkl", allow_pickle=True, encoding="latin1")
    flame = FLAME(deca_cfg.model).to(device)
    flametex = FLAMETex(deca_cfg.model).to(device)
    shape, exp, pose = torch.zeros(1, deca_cfg.model.n_shape), torch.zeros(1, deca_cfg.model.n_exp), torch.zeros(1,
                                                                                                                 deca_cfg.model.n_pose)
    verts, landmarks2d, landmarks3d = flame(shape_params=shape.to(device), expression_params=exp.to(device),
                                            pose_params=pose.to(device))
    faces = flame.faces_tensor.cpu().detach().numpy()

    set_rasterizer(deca_cfg.rasterizer_type)
    render = SRenderY(image_size, obj_filename=deca_cfg.model.topology_path, uv_size=deca_cfg.model.uv_size,
                      rasterizer_type=deca_cfg.rasterizer_type).to(device)

    uvcoords = render.raw_uvcoords[0].cpu().numpy()
    uvfaces = render.uvfaces[0].cpu().numpy()

    # test map
    for key, value in flame_mask.items():
        mapper_set = set()
        for v in value:
            indexes, _ = np.where(faces == v)
            mapper_set = mapper_set.union(set(indexes))

        backboard = np.zeros((image_size, image_size, 3))
        for index in tqdm(mapper_set):
            triangle_vertices = torch.tensor([
                uvcoords[uvfaces[index][0]],
                uvcoords[uvfaces[index][1]],
                uvcoords[uvfaces[index][2]]
                ])
            triangle_vertices = triangle_vertices * image_size
            max_bound, _ = triangle_vertices.max(dim=0)
            max_bound[0], max_bound[1] = int(max_bound[0]), int(max_bound[1])
            min_bound, _ = triangle_vertices.min(dim=0)
            min_bound[0], min_bound[1] = int(min_bound[0]), int(min_bound[1])
            for px in range(int(min_bound[0].item()), int(max_bound[0].item()) + 1):
                for py in range(int(min_bound[1].item()), int(max_bound[1].item()) + 1):
                    if is_in_polygon(triangle_vertices, [px, py]):
                        backboard[image_size - py, px] = 255

        cv2.imwrite(key + '.jpg', backboard)

    backboard = np.zeros((image_size, image_size, 3))
    for uv in uvcoords:
        u, v = uv[0] * image_size, image_size - (uv[1] * image_size)
        backboard = cv2.circle(backboard, (int(u), int(v)), 1, (255, 255, 255), -1)
    cv2.imwrite('total_face.jpg', backboard)


def get_multiview_images(root=r"D:\Creadto\Heritage\Dataset\H3DSv02\h3ds\1141a3328d7e9c4e\image"):
    import mediapipe as mp
    from creadto.models.det import MediaPipeLandmarker

    files = ["img_0001", "img_0024", "img_0052", "img_0038"]
    trans = transforms.ToTensor()
    pivot = 0
    weight = torch.jit.load(r'../_external/pretrained/BasicFacialLandmarker.pth')
    landmarker = MediaPipeLandmarker(model_path='../creadto/_external/mediapipe/face_landmarker.task', visibility=0.5,
                                     presence=0.5)
    image = mp.Image.create_from_file(os.path.join(root, files[pivot]) + ".jpg")
    landmarks = landmarker(image)
    draw = landmarker.draw(image.numpy_view(), landmarks, line=True, sep=False, draw_full=True)
    landmarker.save('./landmarks-' + files[pivot] + '.jpg', draw)


def full_texture_multiview(root=r"D:\Creadto\creadto\dump\crop",
                           mask_root=r"D:\Creadto\creadto\external\creadto\data\texture\mask(head)"):
    image_size = 256
    files = {"front": "img_0001.jpg", "right": "img_0024.jpg", "left": "img_0052.jpg", "back": "img_0038.jpg"}
    rots = {"front": (0, 0, 0), "right": (0, np.pi / 2, 0), "left": (0, -1 * np.pi / 2, 0), "back": (0, np.pi, 0)}
    masks = {"front": "face_eyeball.jpg", "right": "right_ear.jpg", "left": "left_ear.jpg", "back": "scalp.jpg"}
    to_tensor = transforms.ToTensor()
    to_resized = transforms.Resize((image_size, image_size), antialias=True)
    to_pil = ToPILImage()
    encode_dict = proc_vertex(os.path.join(root, files['front']))
    flame = FLAME(deca_cfg.model)
    render = SRenderY(image_size, obj_filename=deca_cfg.model.topology_path, uv_size=deca_cfg.model.uv_size,
                      rasterizer_type=deca_cfg.rasterizer_type)
    verts, landmarks2d, landmarks3d = flame(shape_params=encode_dict['shape'],
                                            expression_params=encode_dict['exp'],
                                            pose_params=encode_dict['pose'])
    cam = encode_dict['cam']
    uv_texture = torch.zeros((1, 3, image_size, image_size))
    for direction in files.keys():
        if direction == "back":
            pvert_test(root = root, mask_root=mask_root, image_name=files[direction], mask_name=masks[direction])
        stub_verts = copy.deepcopy(verts[0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stub_verts.cpu().detach().numpy())
        r = pcd.get_rotation_matrix_from_xyz(rots[direction])
        pcd.rotate(r)
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud('temp.ply', pcd)
        stub_verts = torch.from_numpy(np.asarray(pcd.points)).unsqueeze(dim=0)
        stub_verts = stub_verts.type(torch.FloatTensor)
        trans_verts = util.batch_orth_proj(stub_verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(trans_verts[0].cpu().detach().numpy())
        o3d.visualization.draw_geometries([pcd])

        image = to_resized(to_tensor(Image.open(os.path.join(root, files[direction]))).unsqueeze(dim=0))
        mask = to_tensor(Image.open(os.path.join(mask_root, masks[direction]))).unsqueeze(dim=0)
        mask = F.interpolate(mask, [deca_cfg.model.uv_size, deca_cfg.model.uv_size])
        uv_pverts = render.world2uv(trans_verts)
        dd = uv_pverts.permute(0, 2, 3, 1)[0, :, :, :2]
        uv_gt = F.grid_sample(image, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear', align_corners=False)
        uv_texture = uv_gt[:, :3, :, :] * mask + (uv_texture[:, :3, :, :] * (1 - mask))

        x_uv = uv_pverts.permute(0, 2, 3, 1)[0, :, :, 0]
        x_uv = (x_uv - x_uv.min()) / (x_uv.max() - x_uv.min())
        pil_img = to_pil(x_uv)
        pil_img.save(direction + "-pverts-y.jpg")
        y_uv = uv_pverts.permute(0, 2, 3, 1)[0, :, :, 1]
        y_uv = (y_uv - y_uv.min()) / (y_uv.max() - y_uv.min())
        pil_img = to_pil(y_uv)
        pil_img.save(direction + "-pverts-x.jpg")

        pil_img = to_pil(uv_gt[0])
        pil_img.save(direction + "-uv_gt.jpg")
        pil_img = to_pil(uv_texture[0])
        pil_img.save(direction + "-texture.jpg")


def proc_vertex(path: str):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
    ])
    image = trans(Image.open(path)).unsqueeze(dim=0)
    deca = DECA(config=deca_cfg, device=torch.device("cpu"))
    with torch.no_grad():
        codedict = deca.encode(image)
    return codedict


def pvert_test(root, image_name, mask_root, mask_name):
    to_tensor = transforms.ToTensor()
    to_resized = transforms.Resize((256, 256), antialias=True)
    to_pil = ToPILImage()
    image = to_resized(to_tensor(Image.open(os.path.join(root, image_name))).unsqueeze(dim=0))
    mask = to_tensor(Image.open(os.path.join(mask_root, mask_name))).unsqueeze(dim=0)
    mask[mask >= 0.5] = 1.0
    mask[mask < 0.5] = 0.0
    raw_mask = copy.deepcopy(mask[:, 0])
    mask = mask.permute(0, 2, 3, 1)

    h_pvert = mask[0, :, :, 1]
    hy, hx = torch.where(h_pvert > 0.5)
    for y, x in zip(hy, hx):
        h_pvert[y, x] = y / 256
    h_pvert = h_pvert * 2 - 1
    mask[0, :, :, 1] = h_pvert
    w_pvert = mask[0, :, :, 0]
    wy, wx = torch.where(w_pvert > 0.5)
    for y, x in zip(wy, wx):
        w_pvert[y, x] = x / 145
    w_pvert = w_pvert * 2 - 1
    mask[0, :, :, 0] = w_pvert
    uv_gt = F.grid_sample(image, mask[:, :, :, :2], mode='bilinear', align_corners=False)
    uv_texture = torch.zeros((1, 3, 256, 256))
    uv_texture = uv_gt[:, :3, :, :] * raw_mask + (uv_texture[:, :3, :, :] * (1 - raw_mask))
    pil_img = to_pil(uv_gt[0])
    pil_img.save("temp-back" + "-uv_gt.jpg")
    pil_img = to_pil(uv_texture[0])
    pil_img.save("temp-back" + "-uv_texture.jpg")
    pass


def note():
    from torchvision.transforms import ToPILImage
    trans = ToPILImage()
    pil_img = trans()
    pil_img.save("vis-input_image.jpg")


if __name__ == "__main__":
    make_mask_image()
