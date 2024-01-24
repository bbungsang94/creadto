from torch import optim, nn

from contents.train import get_pack_gat_head
from trainer.gpu import SingleGPURunner
from utils.io import get_loader


def train():
    loader = get_loader()
    kwargs = get_pack_gat_head(batch_size=loader.params['hyperparameters']['batch_size'])
    kwargs.update({
        'loss': loader.get_module('loss', base=nn),
        'optimizer': loader.get_module('optimizer', base=optim, params=kwargs['model'].parameters()),
        'params': loader.params
    })
    gpu_runner = SingleGPURunner(**kwargs)
    gpu_runner.loop()


if __name__ == "__main__":
    train()
