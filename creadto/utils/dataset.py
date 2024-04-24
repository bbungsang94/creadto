import copy
import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from creadto.external.tailor.tailor import GraphTailor


class ImageDataset(Dataset):
    def __init__(self, root: str, pre_load: bool = False):
        """
        :param root: ImagePath
        :param pre_load: [WARNING: Check your RAM size] When this class is instanced, load all images on memory
        """
        files = os.listdir(root)
        self.x = [os.path.join(root, x) for x in files
                  if (".jpg" in x.lower()) or (".png" in x.lower()) or (".jpeg" in x.lower())]

        self.loaded = pre_load
        self.trans = transforms.Compose([transforms.ToTensor()])

        if pre_load:
            for idx, x in enumerate(tqdm(self.x, desc="pre-load processing")):
                self.x[idx] = self.__load__(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.loaded:
            x = self.x[index]
        else:
            x = self.__load__(self.x[index])
        return x

    def __load__(self, path):
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return self.trans(image)


class FlameGraph(Dataset):
    def __init__(self, flame_path, tailor_root, root=None, length=1000, random_seed=241578135, pre_check=False):
        from creadto.external.flame.flame import FLAME
        from easydict import EasyDict
        from creadto.utils.io import load_yaml
        """
        :param root: pth or npy files concluding parameters, landmarks, dimensions
        :param length: data length for training
        """
        torch.manual_seed(random_seed)
        if root:
            # If you have pth or npy files concluding parameters, landmarks, dimensions, load all.
            raise NotImplementedError

        self.length = length
        args = {
            'shape_params': torch.rand(length, 300, dtype=torch.float32) * 4 - 2,
            'expression_params': torch.zeros(length, 100, dtype=torch.float32),
            'pose_params': torch.zeros(length, 6, dtype=torch.float32),
            'jaw': torch.zeros(length, 3, dtype=torch.float32),
            'eyeballs': torch.zeros(length, 6, dtype=torch.float32),
            'tex_params': torch.zeros(length, 50, dtype=torch.float32),
            'camera_params': torch.zeros(length, 3, dtype=torch.float32)
        }
        config = load_yaml(os.path.join(flame_path, 'flame.yaml'))
        config['static_landmark_embedding_path'] = os.path.join(flame_path, config['static_landmark_embedding_path'])
        config['dynamic_landmark_embedding_path'] = os.path.join(flame_path, config['dynamic_landmark_embedding_path'])
        config['flame_lmk_embedding_path'] = os.path.join(flame_path, config['flame_lmk_embedding_path'])
        config['tex_space_path'] = os.path.join(flame_path, config['tex_space_path'])
        config['flame_model_path'] = os.path.join(flame_path, 'generic_model.pkl')
        config = EasyDict(dict(config, **config['constants']))
        
        model = FLAME(config)
        dim_guide = torch.load(os.path.join(tailor_root, 'head_dimension_guide.pt'))
        tailor = GraphTailor(dim_guide)
        self.faces = model.faces_tensor.detach().numpy()

        vertices, _, _ = model(**args)
        tailor.update({'standard': vertices})
        measure = tailor.order(gender=['female'] * length, fast=False, visualize=False)
        args = {
            'shape_params': torch.zeros(length, 300, dtype=torch.float32),
            'expression_params': torch.zeros(length, 100, dtype=torch.float32),
            'pose_params': torch.zeros(length, 6, dtype=torch.float32),
            'jaw': torch.zeros(length, 3, dtype=torch.float32),
            'eyeballs': torch.zeros(length, 6, dtype=torch.float32),
            'tex_params': torch.zeros(length, 50, dtype=torch.float32),
            'camera_params': torch.zeros(length, 3, dtype=torch.float32)
        }
        zero_vertices, _, _ = model(**args)
        tailor.update({'standard': zero_vertices})
        graphs = tailor.to_graph(measure, ["standard"])
        self.x = graphs
        self.y = vertices.detach().clone()

        if pre_check:
            import open3d as o3d
            args = {
                'shape_params': torch.rand(16, 300, dtype=torch.float32) * 4 - 2,
                'expression_params': torch.zeros(16, 100, dtype=torch.float32),
                'pose_params': torch.zeros(16, 6, dtype=torch.float32),
                'jaw': torch.zeros(16, 3, dtype=torch.float32),
                'eyeballs': torch.zeros(16, 6, dtype=torch.float32),
                'tex_params': torch.zeros(16, 50, dtype=torch.float32),
                'camera_params': torch.zeros(16, 3, dtype=torch.float32)
            }

            vertices, _, _ = model(**args)
            for vertex in vertices:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertex.detach().numpy())
                mesh.triangles = o3d.utility.Vector3iVector(self.faces)

                o3d.visualization.draw_geometries([mesh])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class FlameTailor(Dataset):
    def __init__(self, flame_root, tailor_root, length=10000, random_seed=241578135, pre_check=False):
        from creadto.external.flame.flame import FLAME
        """
        :param root: pth or npy files concluding parameters, landmarks, dimensions
        :param length: data length for training
        """
        torch.manual_seed(random_seed)

        self.length = length
        args = {
            'shape_params': torch.concat([
                torch.rand(length-2, 300, dtype=torch.float32) * 2 -1,
                torch.zeros(1, 300, dtype=torch.float32) - 1,
                torch.zeros(1, 300, dtype=torch.float32) + 1
                ]),
            'expression_params': torch.zeros(length, 100, dtype=torch.float32),
            'jaw': torch.zeros(length, 3, dtype=torch.float32),
            'eyeballs': torch.zeros(length, 6, dtype=torch.float32),
            'tex_params': torch.zeros(length, 50, dtype=torch.float32),
            'camera_params': torch.zeros(length, 3, dtype=torch.float32)
        }       
        model = FLAME(root=flame_root)
        dim_guide = torch.load(os.path.join(tailor_root, 'head_dimension_guide.pt'))
        tailor = GraphTailor(dim_guide)
        self.faces = model.faces_tensor.detach().numpy()

        bodies = dict()
        for pose, value in dim_guide['poses'].items():
            temp = value.type(torch.float32)
            temp = temp.expand(args['shape_params'].shape[0], temp.shape[1])
            args['pose_params'] = temp.detach().clone()
            vertices, _, _ = model(**args)
            bodies[pose] = vertices.detach().clone()
        
        # Scaling for diverse data presentation
        #factors = [0.7, 0.8, 0.9, 1, 1.1, 1.2]
        factors = [0.875, 0.9, 0.95, 1.0, 1.05, 1.1, 1.125]
        scale_gap = (length - 2) // len(factors)
        for i, factor in enumerate(factors):
            for pose, value in bodies.items():
                vertices = copy.deepcopy(value)
                for j in range(i * scale_gap, (i+1) * scale_gap):
                    vertices[j, :] = model.scale_to(vertices[j, :], (factor, factor, factor))
                bodies[pose] = vertices
        tailor.update(bodies)
        measure, valid_indices = tailor.dump(gender=['female'] * length, fast=False, normalize=True, visualize=False)

        self.x = measure
        self.y = bodies["standard"][valid_indices].detach().clone()
        self.length = self.x.shape[0]

        if pre_check:
            import open3d as o3d
            vertices = vertices[:8]
            for vertex in vertices:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertex.detach().numpy())
                mesh.triangles = o3d.utility.Vector3iVector(self.faces)

                o3d.visualization.draw_geometries([mesh])
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
class SMPLGraph(Dataset):
    def __init__(self, smpl_path, tailor_root, root=None, length=10000, random_seed=987892, pre_check=False):
        from creadto.external.smpl.smpl import SMPL
        """
        :param root: pth or npy files concluding parameters, landmarks, dimensions
        :param length: data length for training
        """
        torch.manual_seed(random_seed)
        if root:
            # If you have pth or npy files concluding parameters, landmarks, dimensions, load all.
            raise NotImplementedError

        args ={
            'beta': torch.concat([
                torch.zeros(1, 400, dtype=torch.float32) - 2,
                torch.rand(length-2, 400, dtype=torch.float32) * 4 - 2,
                torch.zeros(1, 400, dtype=torch.float32) + 2
                ]),
            'offset': torch.zeros(length, 3, dtype=torch.float32)
        }
        
        self.length = length
        model = SMPL(path=smpl_path)
        self.faces = model.faces
        dim_guide = torch.load(os.path.join(tailor_root, 'body_dimension_guide.pt'))
        tailor = GraphTailor(dim_guide)

        bodies = dict()
        for pose, value in dim_guide['poses'].items():
            temp = value.type(torch.float32)
            temp = temp.expand(args['beta'].shape[0], temp.shape[1], temp.shape[2])
            args['pose'] = temp.detach().clone()
            vertices, _ = model(**args)
            bodies[pose] = vertices.detach().clone()

        tailor.update(bodies)
        gender = "female" if "FEMALE" in smpl_path else "male"
        measure = tailor.order(gender=[gender] * length, fast=False, normalize=False, visualize=False)
        max_vals, _ = torch.max(measure, dim=0, keepdim=True)
        min_vals, _ = torch.min(measure, dim=0, keepdim=True)
        measure = -1 + 2 * (measure - min_vals) / (max_vals - min_vals)
        args ={
            'beta': torch.zeros(length, 400, dtype=torch.float32),
            'offset': torch.zeros(length, 3, dtype=torch.float32)
        }
        zero_bodies = dict()
        for pose, value in dim_guide['poses'].items():
            temp = value.type(torch.float32)
            temp = temp.expand(args['beta'].shape[0], temp.shape[1], temp.shape[2])
            args['pose'] = temp.detach().clone()
            z_vertices, _ = model(**args)

            zero_bodies[pose] = z_vertices.detach().clone()
        tailor.update(zero_bodies)
        graphs = tailor.to_graph(measure, zero_bodies.keys(), normalize=False)
        
        label_vertex = bodies["t"].detach().clone()
        for i in range(label_vertex.shape[0]):
            z_vertex = label_vertex[i]
            z_gap = z_vertex.max(dim=0, keepdim=True)[0] - z_vertex.min(dim=0, keepdim=True)[0]
            z_max_gap = z_gap.max()
            label_vertex[i] = -1 + 2 * (z_vertex - z_vertex.min(dim=0, keepdim=True)[0]) / z_max_gap
        self.x = graphs
        self.y = label_vertex
        
        if pre_check:
            import open3d as o3d
            vertices = zero_bodies['t'][:8]
            for vertex in vertices:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertex.detach().numpy())
                mesh.triangles = o3d.utility.Vector3iVector(model.faces)

                o3d.visualization.draw_geometries([mesh])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class SMPLTailor(Dataset):
    def __init__(self, smpl_root, tailor_root, gender, root=None, length=10000, random_seed=45124, pre_check=False):
        from creadto.external.smpl.smpl import SMPL
        torch.manual_seed(random_seed)
        if root:
            # If you have pth or npy files concluding parameters, landmarks, dimensions, load all.
            raise NotImplementedError

        args ={
            'beta': torch.concat([
                torch.rand(length-2, 400, dtype=torch.float32) * 4 - 2,
                torch.zeros(1, 400, dtype=torch.float32) - 2,
                torch.zeros(1, 400, dtype=torch.float32) + 2
                ]),
            'offset': torch.zeros(length, 3, dtype=torch.float32)
        }
        
        model = SMPL(root=smpl_root, gender=gender)
        self.faces = model.faces
        dim_guide = torch.load(os.path.join(tailor_root, 'body_dimension_guide.pt'))
        tailor = GraphTailor(dim_guide)

        bodies = dict()
        for pose, value in dim_guide['poses'].items():
            temp = value.type(torch.float32)
            temp = temp.expand(args['beta'].shape[0], temp.shape[1], temp.shape[2])
            args['pose'] = temp.detach().clone()
            vertices, _ = model(**args)
            bodies[pose] = vertices.detach().clone()

        # Scaling for diverse data presentation
        factors = [0.7, 0.8, 0.9, 1, 1.1, 1.2]
        scale_gap = (length - 2) // len(factors)
        for i, factor in enumerate(factors):
            for pose, value in bodies.items():
                vertices = copy.deepcopy(value)
                for j in range(i * scale_gap, (i+1) * scale_gap):
                    vertices[j, :] = model.scale_to(vertices[j, :], (factor, factor, factor))
                bodies[pose] = vertices
        tailor.update(bodies)
        gender = "female" if "FEMALE" in smpl_root else "male"
        measure, valid_indices = tailor.dump(gender=[gender] * length, fast=False, normalize=True, visualize=False)
        
        #label_vertex = model.get_parts('body', bodies["t"].detach().clone())
        # for i in range(label_vertex.shape[0]):
        #     z_vertex = label_vertex[i]
        #     z_gap = z_vertex.max(dim=0, keepdim=True)[0] - z_vertex.min(dim=0, keepdim=True)[0]
        #     z_max_gap = z_gap.max()
        #     label_vertex[i] = -1 + 2 * (z_vertex - z_vertex.min(dim=0, keepdim=True)[0]) / z_max_gap
        self.x = measure
        self.y = bodies["t"][valid_indices].detach().clone()
        self.length = self.x.shape[0]

        if pre_check:
            import open3d as o3d
            vertices = bodies['t'][:8]
            for vertex in vertices:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertex.detach().numpy())
                mesh.triangles = o3d.utility.Vector3iVector(model.faces)

                o3d.visualization.draw_geometries([mesh])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]