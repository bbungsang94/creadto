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


def extract_shadow():
    import cv2
    import shutil
    root = r"/workspace/sample/texture-model"
    files = os.listdir(root)
    files = [x for x in files if ".png" in x]
    
    save_root = r"/workspace/sample/removal_images"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)
    
    for file in files:
        bgr = cv2.imread(os.path.join(root, file))
        red = bgr[:, :, -1]
        cv2.imwrite(os.path.join(save_root, file), red)
        
    
if __name__ == "__main__":
    import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TORCH_USE_CUDA_DSA"] = '1'
    
    from example.run_texture import procedure
    procedure("/workspace/sample")
    
    extract_shadow()
    # train()
