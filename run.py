import torch
from torch import optim, nn
from creadto.trainer.gpu import SingleGPURunner
from creadto.utils.io import get_loader


def train():
    from contents.pack import get_pack_dim_head
    loader = get_loader()
    kwargs = get_pack_dim_head(batch_size=loader.params['hyperparameters']['batch_size'],
                               shuffle=loader.params['hyperparameters']['shuffle'],
                               num_workers=loader.params['task']['num_workers'])
    del kwargs['faces']
    kwargs.update({
        'loss': loader.get_module('loss', base=nn),
        'optimizer': loader.get_module('optimizer', base=optim, params=kwargs['model'].parameters()),
        'params': loader.params
    })
    gpu_runner = SingleGPURunner(**kwargs)
    gpu_runner.loop()


def demo():
    from contents.demo import demo_gat_head
    demo_gat_head()


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        pass
 
    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings 
    
    
if __name__ == "__main__":
    from example.body_feature import run
    run()
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # train()
