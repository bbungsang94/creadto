import os
from utils.io import load_yaml
from viewer.base import Concrete
from torch.utils.data import random_split


def get_pack_gat_head(batch_size=1, shuffle=False):
    # 확실히 partial로 진행해도 되는 펑션임
    # prepare config file
    root = './external/flame/'
    file = load_yaml(os.path.join(root, 'flame.yaml'))
    # load instance
    from data.dataset import FlameParameter
    from data.dataloader import FlameGraph
    from models.recon import HeadGATDecoder
    dataset = FlameParameter(**file['constants'])
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = FlameGraph(dataset=train_dataset, flame_path='./external/flame', tailor_path='./external/tailor',
                              batch_size=batch_size, shuffle=shuffle)
    eval_loader = FlameGraph(dataset=eval_dataset, flame_path='./external/flame', tailor_path='./external/tailor',
                             batch_size=batch_size, shuffle=shuffle)
    train_features, train_labels = next(iter(train_loader))
    model = HeadGATDecoder(n_of_node=train_features.num_nodes // train_features.num_graphs,
                           node_dim=train_features.num_node_features // train_features.num_graphs,
                           edge_dim=train_features.num_edge_features // train_features.num_graphs,
                           output_dim=5023)
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'viewer': Concrete()
    }


def get_pack_gat_body(batch_size=1, shuffle=False):
    # 확실히 partial로 진행해도 되는 펑션임
    # prepare config file
    from data.dataset import SMPLParamter
    from data.dataloader import SMPLGraph
    from models.recon import BodyGATDecoder
    # shape 400, pose 3, 55, trans 3
    dataset = SMPLParamter()
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = SMPLGraph(dataset=train_dataset, smpl_path='./external/smpl/SMPLX_NEUTRAL.pkl',
                             tailor_path='./external/tailor', batch_size=batch_size, shuffle=shuffle)
    eval_loader = SMPLGraph(dataset=eval_dataset, smpl_path='./external/smpl/SMPLX_NEUTRAL.pkl',
                            tailor_path='./external/tailor', batch_size=batch_size, shuffle=shuffle)
    train_features, train_labels = next(iter(train_loader))
    model = BodyGATDecoder(n_of_node=train_features.num_nodes // train_features.num_graphs,
                           node_dim=train_features.num_node_features // train_features.num_graphs,
                           edge_dim=train_features.num_edge_features // train_features.num_graphs,
                           output_dim=5023)
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'viewer': Concrete()
    }

def get_pack_head_texture(landmarker, batch_size=1, shuffle=False):
    from data.dataset import ImageDataset
    from torch.utils.data import DataLoader
    from models.det import BasicFacialLandmarker

    dataset = ImageDataset(pre_load=True, root=r"D:\Creadto\Heritage\Dataset\GAN dataset\samples")
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    kwargs = {'batch_size': batch_size, 'shuffle': shuffle}
    train_loader, eval_loader = DataLoader(dataset=train_dataset, **kwargs), DataLoader(dataset=eval_dataset, **kwargs)
    model = BasicFacialLandmarker(landmarker)

    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'viewer': Concrete()
    }