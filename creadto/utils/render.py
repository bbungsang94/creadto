import numpy as np
import pyrender
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
from easydict import EasyDict
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes
from torchvision.ops import masks_to_boxes

from creadto.utils.camera import OrthographicCamera
from creadto.utils.illumination.shader import SoftPhongShader
from creadto.utils.io import save_mesh


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.

    Notice:
        x,y,z are in image space
    """

    def __init__(self, image_size=224):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = EasyDict(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vetices = vertices.clone()
        fixed_vetices[..., :2] = -fixed_vetices[..., :2]
        meshes_screen = Meshes(verts=fixed_vetices.float(), faces=faces.long())
        raster_settings = self.raster_settings

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1  # []
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        # import ipdb; ipdb.set_trace()
        return pixel_vals


class Renderer(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        from pytorch3d.io import load_obj
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coordsw
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = self.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors
        colors = torch.tensor([74, 120, 168])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = self.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        # lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))])
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        """
        vertices: [N, V, 3], vertices in work space, for calculating normals, then shading
        transformed_vertices: [N, V, 3], range(-1, 1), projected vertices, for rendering
        """
        batch_size = vertices.shape[0]
        # rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = self.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = self.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = self.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = self.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = self.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))

        # render
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(),
                                face_vertices.detach(), face_normals.detach()], -1)
        # import ipdb;ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]

        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # remove inner mouth region
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        if lights is not None:
            normal_images = rendering[:, 9:12, :, :].detach()
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  lights)
                    shading_images = shading.reshape(
                        [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)
                    shading_images = shading_images.mean(1)
                else:
                    shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                    shading_images = shading.reshape(
                        [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)
                    shading_images = shading_images.mean(1)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.

        outputs = {
            'images': images * alpha_images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals
        }

        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        """
            sh_coeff: [bz, 9, 3]
        """
        ni = normal_images
        sh = torch.stack([
            ni[:, 0] * 0. + 1., ni[:, 0], ni[:, 1], ni[:, 2], ni[:, 0] * ni[:, 1], ni[:, 0] * ni[:, 2],
            ni[:, 1] * ni[:, 2], ni[:, 0] ** 2 - ni[:, 1] ** 2, 3 * (ni[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        # import ipdb; ipdb.set_trace()
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def add_pointlight(self, vertices, normals, lights):
        """
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        """
        light_positions = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        """
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nlgiht, nv, 3]
        """
        light_direction = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading

    def render_shape(self, vertices, transformed_vertices, images=None, lights=None):
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor([[-0.1, -0.1, 0.2],
                                            [0, 0, 1]]
                                           )[None, :, :].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        # Attributes
        face_vertices = self.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = self.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = self.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = self.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = self.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        # render
        attributes = torch.cat(
            [self.face_colors.expand(batch_size, -1, -1, -1), transformed_face_normals.detach(), face_vertices.detach(),
             face_normals.detach()], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        # albedo
        albedo_images = rendering[:, :3, :, :]
        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        if lights.shape[1] == 9:
            shading_images = self.add_SHlight(normal_images, lights)
        else:
            print('directional')
            shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)

            shading_images = shading.reshape(
                [batch_size, lights.shape[1], albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 1, 4, 2, 3)
            shading_images = shading_images.mean(1)
        images = albedo_images * shading_images

        return images

    def render_normal(self, transformed_vertices, normals):
        """
        -- rendering normal
        """
        batch_size = normals.shape[0]

        # Attributes
        attributes = self.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        """
        sample vertices from world space to uv space
        uv_vertices: [bz, 3, h, w]
        """
        batch_size = vertices.shape[0]
        face_vertices = self.face_vertices(vertices, self.faces.expand(batch_size, -1, -1)).clone().detach()
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]

        return uv_vertices

    def save_obj(self, filename, vertices, textures):
        """
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        """
        save_mesh(filename, vertices, self.faces[0], textures=textures, uvcoords=self.raw_uvcoords[0],
                  uvfaces=self.uvfaces[0])

    @staticmethod
    def face_vertices(vertices, faces):
        """
        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of faces, 3, 3]
        """
        assert (vertices.ndimension() == 3)
        assert (faces.ndimension() == 3)
        assert (vertices.shape[0] == faces.shape[0])
        assert (vertices.shape[2] == 3)
        assert (faces.shape[2] == 3)

        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device
        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
        vertices = vertices.reshape((bs * nv, 3))
        # pytorch only supports long and byte tensors for indexing
        return vertices[faces.long()]

    @staticmethod
    def vertex_normals(vertices, faces):
        """
        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of vertices, 3]
        """
        assert (vertices.ndimension() == 3)
        assert (faces.ndimension() == 3)
        assert (vertices.shape[0] == faces.shape[0])
        assert (vertices.shape[2] == 3)
        assert (faces.shape[2] == 3)

        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device
        normals = torch.zeros(bs * nv, 3).to(device)

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
        vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

        faces = faces.view(-1, 3)
        vertices_faces = vertices_faces.view(-1, 3, 3)

        normals.index_add_(0, faces[:, 1].long(),
                           torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                       vertices_faces[:, 0] - vertices_faces[:, 1]))
        normals.index_add_(0, faces[:, 2].long(),
                           torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                       vertices_faces[:, 1] - vertices_faces[:, 2]))
        normals.index_add_(0, faces[:, 0].long(),
                           torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                       vertices_faces[:, 2] - vertices_faces[:, 0]))

        normals = F.normalize(normals, eps=1e-6, dim=1)
        normals = normals.reshape((bs, nv, 3))
        # pytorch only supports long and byte tensors for indexing
        return normals


class DifferentiableRenderer(nn.Module):

    def __init__(self, resolution, mode='standard', shading=False, num_channels=3):
        super().__init__()
        self.glctx = dr.RasterizeGLContext()
        self.resolution = resolution
        self.num_channels = num_channels
        self.render_func_texture = render_with_texture
        self.shader = SoftPhongShader()
        self.shading = shading
        if mode == 'bounds':
            self.render_func_texture = render_in_bounds_with_texture

    def render_with_texture_map(self, vertex_positions, triface_indices, uv_coords, uv_indices, texture_image,
                                ranges=None, background=None, resolution=None, vertex_normals=None,
                                vertex_positions_world=None):
        if ranges is None:
            ranges = torch.tensor([[0, triface_indices.shape[0]]]).int()
        if resolution is None:
            resolution = self.resolution
        if self.shading:
            color = render_with_texture_shading(glctx=self.glctx, shader=self.shader, vertex_positions=vertex_positions,
                                                triface_indices=triface_indices,
                                                uv_coords=uv_coords, uv_indices=uv_indices, resolution=resolution,
                                                ranges=ranges, vertex_normals=vertex_normals,
                                                vertex_positions_world=vertex_positions_world, background=background)
        else:
            color = self.render_func_texture(self.glctx, vertex_positions, triface_indices, uv_coords, texture_image,
                                             uv_indices, resolution, ranges, background)
        return color[:, :, :, :self.num_channels]


class AbstractRenderer(object):
    def __init__(self, faces=None, img_size=224, use_raymond_lighting=True):
        super(AbstractRenderer, self).__init__()

        self.img_size = img_size
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_size,
            viewport_height=img_size,
            point_size=1.0)
        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.0, 0.0, 0.0))
        if use_raymond_lighting:
            light_nodes = self._create_raymond_lights()
            for node in light_nodes:
                self.scene.add_node(node)

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes

    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def create_mesh(
            self,
            vertices,
            faces,
            color=(0.3, 0.3, 0.3, 1.0),
            wireframe: bool = False,
            deg: float = 0,
            face_colors=None,
            vertex_colors=None,
    ) -> pyrender.Mesh:

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        if face_colors is not None:
            face_colors = np.mean(face_colors, axis=1)
        #  mesh = self.mesh_constructor(vertices, faces, process=False,
        #  face_colors=face_colors)

        curr_vertices = vertices.copy()
        mesh = self.mesh_constructor(
            curr_vertices, faces,
            process=False,
            face_colors=face_colors,
            vertex_colors=vertex_colors)
        if deg != 0:
            rot = self.transf(
                np.radians(deg), [0, 1, 0],
                point=np.mean(curr_vertices, axis=0))
            mesh.apply_transform(rot)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material, smooth=False)

    def update_mesh(self, vertices, faces, body_color=(1.0, 1.0, 1.0, 1.0),
                    deg=0, face_colors=None, vertex_colors=None, ):
        for node in self.scene.get_nodes():
            if node.name == 'body_mesh':
                self.scene.remove_node(node)
                break

        body_mesh = self.create_mesh(
            vertices, faces, color=body_color, deg=deg,
            face_colors=face_colors, vertex_colors=vertex_colors)
        self.scene.add(body_mesh, name='body_mesh')


class WeakPerspectiveCamera(pyrender.Camera):
    PIXEL_CENTER_OFFSET = 0.5

    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=pyrender.camera.DEFAULT_Z_FAR,
                 scale_first=False,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation
        self.scale_first = scale_first

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale
        P[1, 1] = self.scale
        if self.scale_first:
            P[0, 3] = self.translation[0]
            P[1, 3] = -self.translation[1]
        else:
            P[0, 3] = self.translation[0] * self.scale
            P[1, 3] = -self.translation[1] * self.scale
        P[2, 2] = -1

        return P


class OverlayRenderer(AbstractRenderer):
    def __init__(self, faces=None, img_size=224, tex_size=1):
        super(OverlayRenderer, self).__init__(faces=faces, img_size=img_size)

    def update_camera(self, scale, translation, scale_first=False):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = WeakPerspectiveCamera(scale, translation,
                                   scale_first=scale_first,
                                   znear=1e-5,
                                   zfar=1000)
        camera_pose = np.eye(4)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self, vertices, faces,
                 camera_scale, camera_translation, bg_imgs=None,
                 deg=0,
                 return_with_alpha=False,
                 body_color=None,
                 scale_first=False,
                 **kwargs):

        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(camera_scale):
            camera_scale = camera_scale.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = [1.0, 1.0, 0.9]

            if bg_imgs is not None:
                _, H, W = bg_imgs[bidx].shape
                # Update the renderer's viewport
                self.renderer.viewport_height = H
                self.renderer.viewport_width = W

            self.update_camera(camera_scale[bidx], camera_translation[bidx],
                               scale_first=scale_first)
            self.update_mesh(vertices[bidx], faces, body_color=body_color,
                             deg=deg)

            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if bg_imgs is None:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
            else:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
        return np.stack(output_imgs, axis=0)


class HDRenderer(OverlayRenderer):
    def __init__(self, znear: float = 1.0e-2, zfar: float = 1000):
        super(HDRenderer, self).__init__()
        self.znear = znear
        self.zfar = zfar

    def update_camera(self, focal_length, translation, center):
        for node in self.scene.get_nodes():
            if node.name == 'camera':
                self.scene.remove_node(node)

        pc = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=center[0],
            cy=center[1],
            znear=self.znear,
            zfar=self.zfar
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = translation.copy()
        camera_pose[0, 3] *= (-1)
        self.scene.add(pc, pose=camera_pose, name='camera')

    @torch.no_grad()
    def __call__(self,
                 vertices,
                 faces,
                 focal_length,
                 camera_translation,
                 camera_center,
                 bg_imgs,
                 render_bg: bool = True,
                 deg: float = 0,
                 return_with_alpha: bool = False,
                 body_color=None,
                 face_colors=None,
                 vertex_colors=None,
                 **kwargs):
        '''
            Parameters
            ----------
            vertices: BxVx3, torch.Tensor
                The torch Tensor that contains the current vertices to be drawn
            faces: Fx3, np.array
                The faces of the meshes to be drawn. Right now only support a
                batch of meshes with the same topology
            focal_length: B, torch.Tensor
                The focal length used by the perspective camera
            camera_translation: Bx3, torch.Tensor
                The translation of the camera estimated by the network
            camera_center: Bx2, torch.Tensor
                The center of the camera in pixels
            bg_imgs: np.ndarray
                Optional background images used for overlays
            render_bg: bool, optional
                Render on top of the background image
            deg: float, optional
                Degrees to rotate the mesh around itself. Used to render the
                same mesh from multiple viewpoints. Defaults to 0 degrees
            return_with_alpha: bool, optional
                Whether to return the rendered image with an alpha channel.
                Default value is False.
            body_color: list, optional
                The color used to render the image.
        '''
        if torch.is_tensor(vertices):
            vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        if torch.is_tensor(focal_length):
            focal_length = focal_length.detach().cpu().numpy()
        if torch.is_tensor(camera_translation):
            camera_translation = camera_translation.detach().cpu().numpy()
        if torch.is_tensor(camera_center):
            camera_center = camera_center.detach().cpu().numpy()
        if face_colors is not None and torch.is_tensor(face_colors):
            face_colors = face_colors.detach().cpu().numpy()
        if vertex_colors is not None and torch.is_tensor(vertex_colors):
            vertex_colors = vertex_colors.detach().cpu().numpy()
        batch_size = vertices.shape[0]

        output_imgs = []
        for bidx in range(batch_size):
            if body_color is None:
                body_color = [1.0, 1.0, 0.9]

            _, H, W = bg_imgs[bidx].shape
            # Update the renderer's viewport
            self.renderer.viewport_height = H
            self.renderer.viewport_width = W

            self.update_camera(
                focal_length=focal_length[bidx],
                translation=camera_translation[bidx],
                center=camera_center[bidx],
            )
            face_color = None
            if face_colors is not None:
                face_color = face_colors[bidx]
            vertex_color = None
            if vertex_colors is not None:
                vertex_color = vertex_colors[bidx]
            self.update_mesh(
                vertices[bidx], faces, body_color=body_color, deg=deg,
                face_colors=face_color,
                vertex_colors=vertex_color,
            )
            flags = (pyrender.RenderFlags.RGBA |
                     pyrender.RenderFlags.SKIP_CULL_FACES)
            color, depth = self.renderer.render(self.scene, flags=flags)
            color = np.transpose(color, [2, 0, 1]).astype(np.float32) / 255.0
            color = np.clip(color, 0, 1)

            if render_bg:
                if return_with_alpha:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    if bg_imgs[bidx].shape[0] < 4:
                        curr_bg_img = np.concatenate(
                            [bg_imgs[bidx],
                             np.ones_like(bg_imgs[bidx, [0], :, :])
                             ], axis=0)
                    else:
                        curr_bg_img = bg_imgs[bidx]

                    output_img = (color * valid_mask +
                                  (1 - valid_mask) * curr_bg_img)
                    output_imgs.append(np.clip(output_img, 0, 1))
                else:
                    valid_mask = (color[3] > 0)[np.newaxis]

                    output_img = (color[:-1] * valid_mask +
                                  (1 - valid_mask) * bg_imgs[bidx])
                    output_imgs.append(np.clip(output_img, 0, 1))
            else:
                if return_with_alpha:
                    output_imgs.append(color)
                else:
                    output_imgs.append(color[:-1])
        return np.stack(output_imgs, axis=0)


def render_with_texture(glctx, pos_clip, pos_idx, uv_coords, tex_image, col_idx, resolution, ranges, background=None,
                        vertex_normals=None):
    render_resolution = int(resolution)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[render_resolution, render_resolution],
                                         ranges=ranges)
    texc, texd = dr.interpolate(uv_coords[None, ...], rast_out, col_idx, rast_db=rast_out_db, diff_attrs='all')
    # color = dr.texture(tex_image[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=9)
    color = dr.texture(tex_image.permute(0, 2, 3, 1).contiguous(), texc, texd, filter_mode='linear-mipmap-linear',
                       max_mip_level=9)
    mask = rast_out[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color = torch.cat((color, mask.float()), dim=-1)
    return color[:, :, :, :-1]


def render_in_bounds_with_texture(glctx, pos_clip, pos_idx, uv_coords, tex_image, col_idx, resolution, ranges,
                                  background=None, vertex_normals=None):
    render_resolution = int(resolution * 1.2)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[render_resolution, render_resolution],
                                         ranges=ranges)
    texc, texd = dr.interpolate(uv_coords[None, ...], rast_out, col_idx, rast_db=rast_out_db, diff_attrs='all')
    color = dr.texture(tex_image[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=9)
    mask = rast_out[..., -1:] == 0
    if background is None:
        one_tensor = torch.ones((color.shape[0], color.shape[3], 1, 1), device=color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, color)  # [:, :, :, :-1]
    color = torch.cat([color, mask], dim=-1)
    color_crops = []
    boxes = masks_to_boxes(torch.logical_not(mask.squeeze(-1)))
    for img_idx in range(color.shape[0]):
        x1, y1, x2, y2 = [int(val) for val in boxes[img_idx, :].tolist()]
        color_crop = color[img_idx, y1: y2, x1: x2, :].permute((2, 0, 1))
        pad = [[0, 0], [0, 0]]
        if y2 - y1 > x2 - x1:
            total_pad = (y2 - y1) - (x2 - x1)
            pad[0][0] = total_pad // 2
            pad[0][1] = total_pad - pad[0][0]
            pad[1][0], pad[1][1] = 0, 0
            additional_pad = int((y2 - y1) * 0.1)
        else:
            total_pad = (x2 - x1) - (y2 - y1)
            pad[0][0], pad[0][1] = 0, 0
            pad[1][0] = total_pad // 2
            pad[1][1] = total_pad - pad[1][0]
            additional_pad = int((x2 - x1) * 0.1)
        for i in range(4):
            pad[i // 2][i % 2] += additional_pad

        padded = torch.ones((color_crop.shape[0], color_crop.shape[1] + pad[1][0] + pad[1][1],
                             color_crop.shape[2] + pad[0][0] + pad[0][1]), device=color_crop.device)
        padded[:3, :, :] = padded[:3, :, :] * one_tensor[img_idx, :3, :, :]
        padded[:, pad[1][0]: padded.shape[1] - pad[1][1], pad[0][0]: padded.shape[2] - pad[0][1]] = color_crop
        color_crop = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(resolution, resolution),
                                                     mode='bilinear', align_corners=False).permute((0, 2, 3, 1))
        color_crops.append(color_crop)
    return torch.cat(color_crops, dim=0)


def render_with_texture_shading(glctx, shader, vertex_positions, triface_indices, uv_coords, uv_indices, resolution,
                                ranges,
                                background=None, vertex_normals=None, vertex_positions_world=None):
    render_resolution = int(resolution)
    rast_out, rast_out_db = dr.rasterize(glctx, vertex_positions, triface_indices,
                                         resolution=[render_resolution, render_resolution], ranges=ranges)
    mask = rast_out[..., -1:] == 0
    # Interpolate the required attributes
    texc, texd = dr.interpolate(uv_coords[None, ...], rast_out, uv_indices, rast_db=rast_out_db,
                                diff_attrs='all')  # Interpolates the UV Coordinates
    normalsc, normalsd = dr.interpolate(vertex_normals[None, ...], rast_out, triface_indices, rast_db=rast_out_db,
                                        diff_attrs='all')  # Interpolates vertex normals
    vertexc, vertexd = dr.interpolate(vertex_positions_world[None, ...], rast_out, triface_indices, rast_db=rast_out_db,
                                      diff_attrs='all')  # Interpolates world space vertices

    # Apply Shading
    device = texc.device
    shader.set_device(device)

    # The colors are in rang [0,1], so we need to normalize them accordingly
    ambient_color, diffuse_color, specular_color = shader(points=vertexc, normals=normalsc,
                                                          camera_position=torch.tensor([0., 0., -5.])[None, ...].to(
                                                              device))

    if background is None:
        one_tensor = torch.ones((diffuse_color.shape[0], diffuse_color.shape[3], 1, 1), device=diffuse_color.device)
    else:
        one_tensor = background
    one_tensor_permuted = one_tensor.permute((0, 2, 3, 1)).contiguous()
    color = torch.where(mask, one_tensor_permuted, diffuse_color.float())  # [:, :, :, :-1]
    color = torch.cat((color, mask.float()), dim=-1)
    return color[:, :, :, :-1]


def get_orthographic_view(pos_x=0, pos_y=0, max_len=1):
    shape = (max_len * 2, max_len * 2)
    cam_dist = -5.0
    lookat = (0, 0, 0)
    orthograhic_cam = OrthographicCamera(size=shape, near=2.0, far=5000.0,
                                         position=(pos_x, pos_y, -cam_dist), clear_color=(1, 1, 1, 1),
                                         lookat=lookat, up=(0, 1, 0))
    projection_matrix = torch.from_numpy(orthograhic_cam.projection_mat()).float()
    view_matrix = torch.from_numpy(orthograhic_cam.view_mat()).float()
    return projection_matrix, view_matrix


def transform_pos_mvp(pos, mvp):
    """
        Apply projection and view matrices multiplication to the mesh vertices
    Args:
        pos: Given mesh (N, 3)
        mvp: The combined view-projection matrix to be multiplied with

    Returns:

    """
    # noinspection PyArgumentList
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos.device)], axis=1)  # Adding the w-coordinate
    posh = torch.bmm(posw.unsqueeze(0).expand(mvp.shape[0], -1, -1), mvp.permute((0, 2, 1))).reshape((-1, 4))
    pos = torch.zeros_like(posh)
    pos[:, :3] = posh[:, :3] / posh[:, 3:4]
    pos[:, 3] = 1
    return pos


def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:, :, :2]
    # 'input points must use original range'
    if points_scale:
        assert points_scale[0] == points_scale[1]
        points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
        torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1),
        tform)
    if out_scale:  # h,w of output image size
        trans_points_2d[:, :, 0] = trans_points_2d[:, :, 0] / out_scale[1] * 2 - 1
        trans_points_2d[:, :, 1] = trans_points_2d[:, :, 1] / out_scale[0] * 2 - 1
    trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
    return trans_points
