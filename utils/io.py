import copy
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml

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
    cvt = lambda x, t: [t(y) for y in x]
    mesh_text = Path(mesh_path).read_text().splitlines()
    vertices, indices, uvs, uv_indices = [], [], [], []
    for line in mesh_text:
        if line.startswith("v "):
            vertices.append(cvt(line.split(" ")[1:4], float))
        if line.startswith("vt "):
            uvs.append(cvt(line.split(" ")[1:], float))
        if line.startswith("f "):
            if '/' in line:
                indices.append([int(x.split('/')[0]) - 1 for x in line.split(' ')[1:]])
                uv_indices.append([int(x.split('/')[1]) - 1 for x in line.split(' ')[1:]])
            else:
                indices.append([int(x) - 1 for x in line.split(' ')[1:]])
    return np.array(vertices), np.array(indices), np.array(uvs), np.array(uv_indices)


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
