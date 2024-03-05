from torch import optim, nn
from trainer.gpu import SingleGPURunner
from utils.io import get_loader


def train():
    from contents.pack import get_pack_gat_body
    loader = get_loader()
    kwargs = get_pack_gat_body(batch_size=loader.params['hyperparameters']['batch_size'],
                               shuffle=loader.params['hyperparameters']['shuffle'])
    kwargs.update({
        'loss': loader.get_module('loss', base=nn),
        'optimizer': loader.get_module('optimizer', base=optim, params=kwargs['model'].parameters()),
        'params': loader.params
    })
    gpu_runner = SingleGPURunner(**kwargs)
    gpu_runner.loop()


def demo():
    from contents.demo import demo_check_flame_mask
    demo_check_flame_mask()


if __name__ == "__main__":
    train()
