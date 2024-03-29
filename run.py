from torch import optim, nn
from trainer.gpu import SingleGPURunner
from utils.io import get_loader


def train():
    from contents.pack import get_pack_gat_body
    loader = get_loader()
    kwargs = get_pack_gat_body(batch_size=loader.params['hyperparameters']['batch_size'],
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
    from contents.demo import demo_get_body
    demo_get_body("female")


if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
