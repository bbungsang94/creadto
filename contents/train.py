from viewer.base import Concrete


def get_pack_gat_head(batch_size=1):
    # prepare config file
    import os
    from utils.io import load_yaml
    from torch.utils.data import random_split
    root = './external/flame/'
    file = load_yaml(os.path.join(root, 'flame.yaml'))
    # load instance
    from data.dataset import FlameParameter
    from data.dataloader import FlameGraph
    from models.recon import HeadGATDecoder
    dataset = FlameParameter(**file['constants'])
    train_dataset, eval_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = FlameGraph(dataset=train_dataset, flame_path='./external/flame', tailor_path='./external/tailor',
                              batch_size=batch_size, shuffle=False)
    eval_loader = FlameGraph(dataset=eval_dataset, flame_path='./external/flame', tailor_path='./external/tailor',
                             batch_size=batch_size, shuffle=False)
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
