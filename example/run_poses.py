import os
import torch
import os.path as osp
import shutil

def load_texture_model(root):
    from creadto.utils.io import load_mesh, load_image
    files = os.listdir(root)
    file_dict = dict()
    file_dict['name'] = [x for x in files if ".obj" in x]
    file_dict['full_path'] = [osp.join(root, x) for x in file_dict['name']]
    file_dict['plain_model'] = []
    file_dict['texture'] = []
    for file in file_dict['full_path']:
        v, f, uvcoords, uvfaces = load_mesh(file)
        texture = load_image(file.replace('.obj', '.png'), integer=False)
        file_dict['plain_model'].append((v, f, uvcoords, uvfaces))
        file_dict['texture'].append(texture)
    return file_dict

def load_poses_parameters(root):
    files = os.listdir(root)
    file_dict = {
        'pose_parameter': [],
        'pose_name': files,
    }
    for file in files:
        pose = torch.load(osp.join(root, file))
        file_dict['pose_parameter'].append(pose)
    return file_dict
    
def procedure(root):
    from tqdm import tqdm
    from creadto.utils.io import save_mesh
    from creadto._external.smpl.smpl import SMPL
    from torchvision.transforms.functional import to_pil_image
    
    file_dict = load_texture_model(osp.join(root, "texture-model"))
    file_dict.update(load_poses_parameters(osp.join(root, "pose_parameter")))
    smpl = SMPL("./creadto-model")
    
    if osp.exists(osp.join(root, "pose-imitating")):
        shutil.rmtree(osp.join(root, "pose-imitating"))
    os.mkdir(osp.join(root, "pose-imitating"))
    
    for i, model_pack in tqdm(enumerate(file_dict['plain_model'])):
        folder = file_dict['name'][i].replace('.obj', '')
        new_root = osp.join(root, "pose-imitating", folder)
        os.mkdir(new_root)

        v, f, uvcoords, uvfaces = model_pack
        for j, pose in enumerate(file_dict['pose_parameter']):
            texture = file_dict['texture'][i]
            pose_name = file_dict['pose_name'][j].replace('.pth', '')
            new_v = smpl.pose_to(v.unsqueeze(dim=0), pose.unsqueeze(dim=0))
            save_mesh(osp.join(new_root, folder + pose_name + ".obj"), new_v[0], f.cpu().detach().numpy(),
                      texture=to_pil_image(texture[0]), uvcoords=uvcoords, uvfaces=uvfaces)
