import torch
from easydict import EasyDict

from external.flame.flame import FLAME, FLAMETex
from models.gen import StyleGANAda
from models.mlp import MappingNetwork
from utils.io import load_mesh, make_dir
from utils.math import l2_distance


def demo_gat_head():
    import open3d as o3d
    from contents.pack import get_pack_gat_head

    pack = get_pack_gat_head(batch_size=16)
    model_dict = torch.load(r'./external/pretrained/NewHeadDecoder.pth')
    pack['model'].load_state_dict(model_dict['model'])

    train_features, train_labels = next(iter(pack['loaders'][0]))
    pack['model'].eval()
    with torch.no_grad():
        output = pack['model'](train_features)
        face = pack['faces']
        for vertices in output['output']:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(face)

            o3d.visualization.draw_geometries([mesh])


def demo_get_body(gender: str):
    import open3d as o3d
    from contents.pack import get_pack_gat_body

    pack = get_pack_gat_body(batch_size=16, num_workers=0)
    model_dict = torch.load(r'./external/pretrained/BodyDecoder_' + gender + '.pth')
    pack['model'].load_state_dict(model_dict['model'])
    train_features, train_labels = next(iter(pack['loaders'][0]))
    pack['model'].eval()
    with torch.no_grad():
        output = pack['model'](train_features)

        face = pack['faces']
        loss = torch.nn.functional.mse_loss(output['output'], train_labels)
        line = "average loss: %0.5f \n" % loss
        print(line)
        for vertices in output['output']:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(face)

            o3d.visualization.draw_geometries([mesh])

def demo_head_texture():
    from contents.pack import get_pack_head_texture
    model = torch.jit.load(r'./external/pretrained/BasicFacialLandmarker.pth')
    pack = get_pack_head_texture(batch_size=16, shuffle=True, landmarker=model)

    images = next(iter(pack['loaders'][0]))
    output = pack['model'](images)
    # landmarker min bound, max bound -> FLAME min bound, max bound
    # to uv mapping seperated human parts
    pass


def demo_check_flame_mask():
    import os, sys
    import cv2
    import torch
    import torchvision
    import torch.nn.functional as F
    import torch.nn as nn
    import numpy as np
    import datetime
    from utils.render import transform_pos_mvp, transform_points, DifferentiableRenderer, Renderer

    class PhotometricFitting(object):
        def __init__(self, config, device='cuda'):
            self.batch_size = config.batch_size
            self.image_size = config.image_size
            self.config = config
            self.device = device
            #
            self.flame = FLAME(self.config).to(self.device)
            self.flametex = FLAMETex(self.config).to(self.device)

            self._setup_renderer()

        def tensor_vis_landmarks(self, images, landmarks, gt_landmarks=None, color='g', isScale=True):
            # visualize landmarks
            vis_landmarks = []
            images = images.cpu().numpy()
            predicted_landmarks = landmarks.detach().cpu().numpy()
            if gt_landmarks is not None:
                gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
            for i in range(images.shape[0]):
                image = images[i]
                image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy();
                image = (image * 255)
                if isScale:
                    predicted_landmark = predicted_landmarks[i] * image.shape[0] / 2 + image.shape[0] / 2
                else:
                    predicted_landmark = predicted_landmarks[i]

                image_landmarks = self.plot_kpts(image, predicted_landmark, color)
                vis_landmarks.append(image_landmarks)

            vis_landmarks = np.stack(vis_landmarks)
            vis_landmarks = torch.from_numpy(
                vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
            return vis_landmarks

        def _setup_renderer(self):
            mesh_file = './external/flame/head_template.obj'
            self.render = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)

        def optimize(self, images, landmarks, image_masks, savefolder=None):
            bz = images.shape[0]
            shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
            tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
            exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
            pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
            cam = torch.zeros(bz, self.config.camera_params);
            cam[:, 0] = 5.
            cam = nn.Parameter(cam.float().to(self.device))
            lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
            e_opt = torch.optim.Adam(
                [shape, exp, pose, cam, tex, lights],
                lr=self.config.e_lr,
                weight_decay=self.config.e_wd
            )
            e_opt_rigid = torch.optim.Adam(
                [pose, cam],
                lr=self.config.e_lr,
                weight_decay=self.config.e_wd
            )

            gt_landmark = landmarks

            # rigid fitting of pose and camera with 51 static face landmarks,
            # this is due to the non-differentiable attribute of contour landmarks trajectory
            for k in range(200):
                losses = {}
                vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp,
                                                                pose_params=pose)
                trans_vertices = self.batch_orth_proj(vertices, cam);
                trans_vertices[..., 1:] = - trans_vertices[..., 1:]
                landmarks2d = self.batch_orth_proj(landmarks2d, cam);
                landmarks2d[..., 1:] = - landmarks2d[..., 1:]
                landmarks3d = self.batch_orth_proj(landmarks3d, cam);
                landmarks3d[..., 1:] = - landmarks3d[..., 1:]

                losses['landmark'] = l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * config.w_lmks

                all_loss = 0.
                for key in losses.keys():
                    all_loss = all_loss + losses[key]
                losses['all_loss'] = all_loss
                e_opt_rigid.zero_grad()
                all_loss.backward()
                e_opt_rigid.step()

                loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                for key in losses.keys():
                    loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
                if k % 10 == 0:
                    print(loss_info)

                if k % 10 == 0:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        self.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        self.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        self.tensor_vis_landmarks(images[visind], landmarks3d[visind]))

                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

            # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
            for k in range(200, 1000):
                losses = {}
                vertices, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp,
                                                                pose_params=pose)
                trans_vertices = self.batch_orth_proj(vertices, cam);
                trans_vertices[..., 1:] = - trans_vertices[..., 1:]
                landmarks2d = self.batch_orth_proj(landmarks2d, cam);
                landmarks2d[..., 1:] = - landmarks2d[..., 1:]
                landmarks3d = self.batch_orth_proj(landmarks3d, cam);
                landmarks3d[..., 1:] = - landmarks3d[..., 1:]

                losses['landmark'] = l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * config.w_lmks
                losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * config.w_shape_reg  # *1e-4
                losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * config.w_expr_reg  # *1e-4
                losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * config.w_pose_reg

                ## render
                albedos = self.flametex(tex) / 255.
                topilimage = torchvision.transforms.ToPILImage()
                image_pil = topilimage(albedos[0])
                image_pil.save('raw_texture.jpg')

                ops = self.render(vertices, trans_vertices, albedos, lights)
                predicted_images = ops['images']
                losses['photometric_texture'] = (image_masks * (ops['images'] - images).abs()).mean() * config.w_pho

                all_loss = 0.
                for key in losses.keys():
                    all_loss = all_loss + losses[key]
                losses['all_loss'] = all_loss
                e_opt.zero_grad()
                all_loss.backward()
                e_opt.step()

                loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                for key in losses.keys():
                    loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

                if k % 10 == 0:
                    print(loss_info)

                # visualize
                if k % 10 == 0:
                    grids = {}
                    visind = range(bz)  # [0]
                    grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                    grids['landmarks_gt'] = torchvision.utils.make_grid(
                        self.tensor_vis_landmarks(images[visind], landmarks[visind]))
                    grids['landmarks2d'] = torchvision.utils.make_grid(
                        self.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                    grids['landmarks3d'] = torchvision.utils.make_grid(
                        self.tensor_vis_landmarks(images[visind], landmarks3d[visind]))
                    grids['albedoimage'] = torchvision.utils.make_grid(
                        (ops['albedo_images'])[visind].detach().cpu())
                    grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
                    shape_images = self.render.render_shape(vertices, trans_vertices, images)
                    grids['shape'] = torchvision.utils.make_grid(
                        F.interpolate(shape_images[visind], [224, 224])).detach().float().cpu()

                    # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
                    grid = torch.cat(list(grids.values()), 1)
                    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

                    cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

            single_params = {
                'shape': shape.detach().cpu().numpy(),
                'exp': exp.detach().cpu().numpy(),
                'pose': pose.detach().cpu().numpy(),
                'cam': cam.detach().cpu().numpy(),
                'verts': trans_vertices.detach().cpu().numpy(),
                'albedos': albedos.detach().cpu().numpy(),
                'tex': tex.detach().cpu().numpy(),
                'lit': lights.detach().cpu().numpy()
            }
            return single_params

        def run(self, imagepath, landmarkpath):
            # The implementation is potentially able to optimize with images(batch_size>1),
            # here we show the example with a single image fitting
            images = []
            landmarks = []
            image_masks = []

            image_name = os.path.basename(imagepath)[:-4]
            savefile = os.path.sep.join([self.config.savefolder, image_name + '.npy'])

            # photometric optimization is sensitive to the hair or glass occlusions,
            # therefore we use a face segmentation network to mask the skin region out.
            image_mask_folder = './contents/FFHQ_seg/'
            image_mask_path = os.path.sep.join([image_mask_folder, image_name + '.npy'])

            image = cv2.resize(cv2.imread(imagepath), (config.cropped_size, config.cropped_size)).astype(
                np.float32) / 255.
            image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
            images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))

            image_mask = np.load(image_mask_path, allow_pickle=True)
            image_mask = image_mask[..., None].astype('float32')
            image_mask = image_mask.transpose(2, 0, 1)
            image_mask_bn = np.zeros_like(image_mask)
            image_mask_bn[np.where(image_mask != 0)] = 1.
            image_masks.append(torch.from_numpy(image_mask_bn[None, :, :, :]).to(self.device))

            landmark = np.load(landmarkpath).astype(np.float32)
            landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
            landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
            landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))

            images = torch.cat(images, dim=0)
            images = F.interpolate(images, [self.image_size, self.image_size])
            image_masks = torch.cat(image_masks, dim=0)
            image_masks = F.interpolate(image_masks, [self.image_size, self.image_size])

            landmarks = torch.cat(landmarks, dim=0)
            savefolder = os.path.sep.join([self.config.savefolder, image_name])

            make_dir(savefolder)
            # optimize
            single_params = self.optimize(images, landmarks, image_masks, savefolder)
            self.render.save_obj(filename=savefile[:-4] + '.obj',
                                 vertices=torch.from_numpy(single_params['verts'][0]).to(self.device),
                                 textures=torch.from_numpy(single_params['albedos'][0]).to(self.device)
                                 )
            np.save(savefile, single_params)


        @staticmethod
        def plot_kpts(image, kpts, color='r'):
            end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1
            if color == 'r':
                c = (255, 0, 0)
            elif color == 'g':
                c = (0, 255, 0)
            elif color == 'b':
                c = (255, 0, 0)
            image = image.copy()
            kpts = kpts.copy()

            for i in range(kpts.shape[0]):
                st = kpts[i, :2]
                if kpts.shape[1] == 4:
                    if kpts[i, 3] > 0.5:
                        c = (0, 255, 0)
                    else:
                        c = (0, 0, 255)
                image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)
                if i in end_list:
                    continue
                ed = kpts[i + 1, :2]
                image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), 1)

            return image

        @staticmethod
        def batch_orth_proj(X, camera):
            '''
                X is N x num_points x 3
            '''
            camera = camera.clone().view(-1, 1, 3)
            X_trans = X[:, :, :2] + camera[:, :, 1:]
            X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
            shape = X_trans.shape
            # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
            Xn = (camera[:, :, 0:1] * X_trans)
            return Xn

    sys.path.append('./models/')
    torch.backends.cudnn.benchmark = True
    ffhq_root = r"./contents/FFHQ"
    image_name = os.path.join(ffhq_root, "00008.png")
    device_name = "cuda"
    config = {
        # FLAME
        'flame_model_path': './external/flame/generic_model.pkl',  # acquire it from FLAME project page
        'static_landmark_embedding_path': './external/flame/flame_static_embedding.pkl',
        'dynamic_landmark_embedding_path': './external/flame/flame_dynamic_embedding.npy',
        'flame_lmk_embedding_path': './external/flame/landmark_embedding.npy',
        'tex_space_path': './external/flame/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': './test_results',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,

        'constants':
            {
                'shape': 100,
                'expression': 50,
                'jaw': 3,
                'rotation': 3,
                'eyeballs': 6,
                'neck': 3,
                'translation': 3,
                'scale': 1,
            }
    }
    """
    flame_model_path, static_landmark_embedding_path, dynamic_landmark_embedding_path,
                 use_face_contour, constants
    """
    config = EasyDict(config)
    make_dir(config.savefolder)

    config.batch_size = 1
    fitting = PhotometricFitting(config, device=device_name)

    input_folder = ffhq_root

    imagepath = image_name
    landmarkpath = image_name.replace('.png', '.npy')
    fitting.run(imagepath, landmarkpath)


def demo_mlp_texture():
    import torch
    import pickle
    from torchvision.utils import save_image
    from utils.render import get_orthographic_view
    from utils.io import load_yaml
    from utils.render import transform_pos_mvp, transform_points, DifferentiableRenderer, Renderer

    image_size = 512
    deca_size = 224
    batch_size = 4
    custom_scale_factor = 2.0
    custom_scale_factor_image = 1024 / image_size
    projection_matrix, view_matrix = get_orthographic_view(pos_x=0, pos_y=0)

    def process_rendering(vertices, camera, tform, projection_matrix, view_matrix):
        trans_verts = vertices[:, :2] + camera[1:]
        trans_verts = torch.cat([trans_verts, vertices[:, 2:]], dim=1)
        scaled_verts = custom_scale_factor * trans_verts * camera[0]
        vertices_mvp = transform_pos_mvp(scaled_verts, torch.matmul(projection_matrix, view_matrix).to(
            torch.device("cuda:0")).unsqueeze(0))
        points_scale = [deca_size, deca_size]  # Increases scale and shifts right + bottom for large values
        h, w = [custom_scale_factor_image * image_size,
                custom_scale_factor_image * image_size]  # Increases scale and shifts right + bottom for smaller values
        tform_1 = torch.inverse(tform[None, ...]).transpose(1, 2).to(torch.device("cuda:0"))
        vertices_mvp = transform_points(vertices_mvp.unsqueeze(0), tform_1, points_scale, [h, w])[0]
        return vertices_mvp

    def visualize_rendering():

        config = load_yaml('./external/clipface/clipface.yaml')

        # Initialize the models
        flame_params = "external/clipface/flame_params.pkl"
        deca_warped_path = 'external/clipface/sample_deca.pkl'
        tex_mapper_ckpt = 'external/clipface/mona_lisa.pt'  # TODO: Update checkpoint path here
        w_init_pth = 'external/clipface/clip_latents_val/w_001368_5907.pt'  # TODO: Update texture latent code path here
        _flame = FLAME(**config['FLAME'])
        _renderer = DifferentiableRenderer(image_size, mode='standard', num_channels=3, shading=False)

        _G = StyleGANAda(z_dim=config['latent_dim'], w_dim=config['latent_dim'],
                         w_num_layers=config['num_mapping_layers'], img_resolution=config['image_size'],
                         img_channels=3, synthesis_layer=config['generator'])

        _texture_mapper_list = []
        mapper_state_dict = torch.load(tex_mapper_ckpt)
        for level in range(18):
            mapper = MappingNetwork(z_dim=config['latent_dim'], w_dim=config['latent_dim'], num_layers=4)
            mapper.load_state_dict(mapper_state_dict[f"level_{level}"])
            mapper = mapper.to(torch.device("cuda:0"))
            _texture_mapper_list.append(mapper)

        _G.load_state_dict(torch.load(config['pretrained_stylegan_pth'], map_location=torch.device("cpu")))
        _G.eval()
        w_init_code = torch.load(w_init_pth)

        with open(flame_params, 'rb') as f:
            flame_params = pickle.load(f)

        with open(deca_warped_path, 'rb') as f:
            data_dict = pickle.load(f)
        tform = torch.tensor(data_dict['tform'].params).float()

        # Load Mesh
        _, faces, uvs, uv_indices = load_mesh("external/clipface/head_template.obj")
        faces = torch.from_numpy(faces).int()
        uvs = torch.from_numpy(uvs).float()
        uvs = torch.cat([uvs[:, 0:1], 1 - uvs[:, 1:2]], dim=1)
        uv_indices = torch.from_numpy(uv_indices).int()

        # Set device
        _flame = _flame.to(torch.device("cuda:0"))
        _G = _G.to(torch.device("cuda:0"))
        _renderer = _renderer.to(torch.device("cuda:0"))
        w_init_code = w_init_code.to(torch.device("cuda:0")).unsqueeze(0)
        faces = faces.to(torch.device("cuda:0"))
        uvs = uvs.to(torch.device("cuda:0"))
        uv_indices = uv_indices.to(torch.device("cuda:0"))
        camera = torch.from_numpy(flame_params['camera']).float().to(torch.device("cuda:0"))
        shape_params = torch.from_numpy(flame_params['shape']).float().to(torch.device("cuda:0")).unsqueeze(0)
        expression_params = torch.from_numpy(flame_params['expression']).float().to(torch.device("cuda:0")).unsqueeze(0)
        pose_params = torch.from_numpy(flame_params['pose']).float().to(torch.device("cuda:0")).unsqueeze(0)

        with torch.no_grad():
            w_offsets = None
            for idx, mapper in enumerate(_texture_mapper_list):
                w_offset_layer = mapper(w_init_code[:, idx, :])
                if w_offsets is None:
                    w_offsets = w_offset_layer
                else:
                    w_offsets = torch.cat((w_offsets, w_offset_layer), dim=0)
            w_offsets = w_offsets.unsqueeze(0)

            w = w_init_code + w_offsets
            init_texture = _G.synthesis(w_init_code, noise_mode='const')
            predicted_texture = _G.synthesis(w, noise_mode='const')
            pred_vertices = _flame(batch_size=1, shape=shape_params, expression=expression_params)[0].squeeze()
            vertices_mvp = process_rendering(pred_vertices, camera, tform, projection_matrix, view_matrix).contiguous()

            pred_init = _renderer.render_with_texture_map(vertices_mvp, faces, uvs, uv_indices, init_texture,
                                                          background=None).permute(0, 3, 1, 2)
            pred_final = _renderer.render_with_texture_map(vertices_mvp, faces, uvs, uv_indices, predicted_texture,
                                                           background=None).permute(0, 3, 1, 2)
            prediction = torch.cat((pred_init, pred_final), dim=3)
            save_image(prediction, f"prediction.jpg", value_range=(-1, 1), normalize=True)

    visualize_rendering()
