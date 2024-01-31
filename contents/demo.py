import torch

from external.flame.flame import FLAME
from models.gen import StyleGANAda
from models.mlp import MappingNetwork
from utils.io import load_mesh
from utils.render import transform_pos_mvp, transform_points, DifferentiableRenderer


def demo_gat_head():
    import open3d as o3d
    from contents.pack import get_pack_gat_head

    pack = get_pack_gat_head(batch_size=16)
    model_dict = torch.load(r'./external/pretrained/GATDecoder.pth')
    pack['model'].load_state_dict(model_dict['model'])
    train_features, train_labels = next(iter(pack['loaders'][0]))
    output = pack['model'](train_features)

    face = pack['loaders'][0].model.faces
    for vertices in output['output']:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(face)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.detach().numpy())

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


def demo_mlp_texture():
    import torch
    import yaml
    import pickle
    from omegaconf import OmegaConf
    import numpy as np
    import cv2
    from torchvision.utils import save_image
    from utils.render import get_orthographic_view
    from utils.io import load_yaml
    from torchvision.transforms.functional import to_pil_image

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
