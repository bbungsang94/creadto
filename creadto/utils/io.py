import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from PIL.Image import Image
import numpy as np
import torch
import yaml
from skimage.io import imsave

from contents import Contents


def load_yaml(x: str) -> Dict[str, Any]:
    with open(x) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    return config


def read_json(full_path=''):
    if '.json' not in full_path:
        full_path += '.json'

    with open(full_path, "r") as f:
        file = json.load(f)
    return file


def load_mesh(mesh_path):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(mesh_path, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long); faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        faces,
        uvcoords,
        uv_faces
    )


def save_mesh(obj_name,
              vertices,
              faces,
              colors=None,
              texture: Image = None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map: Image = None,
              ):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 0], faces[i, 1], faces[i, 2]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    # out_normal_map = normal_map / (np.linalg.norm(
                    #     normal_map, axis=-1, keepdims=True) + 1e-9)
                    # out_normal_map = (out_normal_map + 1) * 0.5
                    normal_map.save(normal_name)
            texture.save(texture_name)


def get_loader(root='./'):
    parameter = read_json(os.path.join(root, 'default.json'))
    module_loader = ModuleLoader(root=root, params=parameter)
    return module_loader


def save_torch(filename: str, model: torch.nn.Module, mode: str = "state_dict", **kwargs):
    if mode == "jit":
        m = torch.jit.script(model)
        with torch.no_grad():
            m.eval()
            m.save(filename)
        if "BestModel" in filename:
            timeline = datetime.now().strftime('%Y%m%d%H%M%S')
            torch.save(kwargs, filename.replace('BestModel', timeline))
    else:
        kwargs.update({'model': model.state_dict()})
        torch.save(kwargs, filename)


def replace_values(source, target):
    for key in source.keys():
        if key in target:
            source[key] = copy.deepcopy(target[key])
    return source


def make_dir(path):
    if os.path.exists(path) is False:
        dir_q = []
        sub_path = path
        while True:
            directory, folder = os.path.split(sub_path)
            sub_path = directory
            dir_q.append(folder)
            if os.path.exists(directory):
                for target in reversed(dir_q):
                    sub_path = os.path.join(sub_path, target)
                    os.mkdir(os.path.join(sub_path))
                break


def clean_folder(path):
    if not os.path.exists(path):
        return
    folders = os.listdir(path)
    for folder in folders:
        files = os.listdir(os.path.join(path, folder))
        if len(files) <= 1:
            shutil.rmtree(os.path.join(path, folder))


class ModuleLoader:
    def __init__(self, root, params):
        self._root = root
        self.params = params
        self.module = Contents()

    def get_module(self, kind, base, **kwargs):
        key = self.params['modules'][kind]
        if 'metric' == kind:
            kind = 'loss'
        func, args = self.get_safety_registry(kind, key, **kwargs)
        if func is None:
            key = args['name']
            del args['name']
            func = getattr(base, key)(**args)
        return func

    def get_args(self, kind, key, **options):
        args = self.params[kind]
        hyperparameter = self.params['hyperparameters']
        args = replace_values(args, hyperparameter)
        for key in options.keys():
            args[key] = options[key]

        return args

    def get_safety_registry(self, kind, key, **kwargs):
        args = self.get_args(kind, key, **kwargs)
        key = key + '_' + kind
        result = None
        if self.module[key]:
            result = self.module[key](**args)
        return result, args
