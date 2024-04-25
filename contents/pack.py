from creadto.viewer.base import Concrete
from torch.utils.data import random_split


def get_pack_gat_head(batch_size=1, shuffle=False, num_workers=0):
    # 확실히 partial로 진행해도 되는 펑션임
    # load instance
    from creadto.utils.dataset import FlameGraph
    from creadto.utils.dataloader import GraphLoader
    from creadto.models.recon import HeadGATDecoder
    dataset = FlameGraph(flame_path='./_external/flame', tailor_root='./_external/tailor', pre_check=False)
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = GraphLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    eval_loader = GraphLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_features, train_labels = next(iter(train_loader))
    model = HeadGATDecoder(n_of_node=train_features.num_nodes // train_features.num_graphs,
                           node_dim=train_features.num_node_features,
                           edge_dim=train_features.num_edge_features,
                           output_dim=5023)
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'faces': dataset.faces,
        'viewer': Concrete()
    }

def get_pack_dim_head(batch_size=1, shuffle=False, pin_memory=False, num_workers=0):
    from creadto.utils.dataset import FlameTailor
    from creadto.utils.dataloader import TensorLoader
    from creadto.models.recon import BasicDecoder
    dataset = FlameTailor(length=25000, flame_root='./_external/flame',
                         tailor_root='./_external/tailor', pre_check=False)
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = TensorLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                pin_memory=pin_memory, num_workers=num_workers)
    eval_loader = TensorLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle,
                               pin_memory=pin_memory, num_workers=num_workers)
    train_features, train_labels = next(iter(train_loader))
    model = BasicDecoder(input_dim=train_features.shape[-1], output_dim=train_labels.shape[1], n_layers=3)
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'faces': dataset.faces,
        'viewer': Concrete()
    }

def get_pack_gat_head_regressor(batch_size=1, shuffle=False, num_workers=0):
    # 확실히 partial로 진행해도 되는 펑션임
    # load instance
    from creadto.utils.dataset import FlameParameter
    from creadto.utils.dataloader import TensorLoader
    from creadto.models.mlp import BasicRegressor
    dataset = FlameParameter(flame_root='./_external/flame', tailor_root='./_external/tailor', pre_check=False)
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = TensorLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    eval_loader = TensorLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_features, train_labels = next(iter(train_loader))
    model = BasicRegressor(input_dim=train_features.shape[-1], output_dim=train_labels.shape[-1])
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'faces': dataset.faces,
        'viewer': Concrete()
    }

def get_pack_dim_body(gender, batch_size=1, shuffle=False, pin_memory=False, num_workers=0):
    from creadto.utils.dataset import SMPLTailor
    from creadto.utils.dataloader import TensorLoader
    from creadto.models.recon import BasicDecoder
    dataset = SMPLTailor(length=25000, smpl_root='./_external/smpl', gender=gender,
                         tailor_root='./_external/tailor', pre_check=False)
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = TensorLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                pin_memory=pin_memory, num_workers=num_workers)
    eval_loader = TensorLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle,
                               pin_memory=pin_memory, num_workers=num_workers)
    train_features, train_labels = next(iter(train_loader))
    model = BasicDecoder(input_dim=train_features.shape[-1], output_dim=train_labels.shape[1], n_layers=3)
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'faces': dataset.faces,
        'viewer': Concrete()
    }
    
def get_pack_gat_body(batch_size=1, shuffle=False, pin_memory=False, num_workers=0):
    # 확실히 partial로 진행해도 되는 펑션임
    # prepare config file
    from creadto.utils.dataset import SMPLGraph
    from creadto.utils.dataloader import GraphLoader
    from creadto.models.recon import BodyGATDecoder
    # shape 400, pose 3, 55, trans 3
    dataset = SMPLGraph(length=4000, smpl_path='./_external/smpl/SMPLX_FEMALE.pkl', tailor_root='./_external/tailor',
                            pre_check=False)
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = GraphLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                               pin_memory=pin_memory, num_workers=num_workers)
    eval_loader = GraphLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=shuffle,
                              pin_memory=pin_memory, num_workers=num_workers)
    train_features, _ = next(iter(train_loader))
    model = BodyGATDecoder(n_of_node=train_features.num_nodes // train_features.num_graphs,
                           node_dim=train_features.num_node_features,
                           edge_dim=train_features.num_edge_features,
                           output_dim=10475)
    return {
        'model': model,
        'loaders': (train_loader, eval_loader),
        'faces': dataset.faces,
        'viewer': Concrete()
    }


def get_pack_head_texture(landmarker, batch_size=1, shuffle=False):
    from creadto.utils.dataset import ImageDataset
    from torch.utils.data import DataLoader
    from creadto.models.det import BasicFacialLandmarker

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
