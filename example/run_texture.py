import os
import os.path as osp
import shutil
import torchvision
import numpy as np
from PIL import Image


def image_to_texture(root, mini_batch_size):
    from creadto.models.tex import PaintHuman
    
    model = PaintHuman()
    files = os.listdir(root)
    pack = []
    for iter in range(int(len(files) / mini_batch_size + 0.9)):
        mini_batch = files[iter * mini_batch_size:(iter + 1) * mini_batch_size]
        raw_images = []
        for name in mini_batch:
            image = Image.open(osp.join(root, name))
            raw_images.append(image)
        pack.append((model(raw_images), [x.split('.')[0] for x in mini_batch]))
    return pack

def load_plane_models(root, names):
    from creadto.utils.io import load_mesh
    models = dict()
    for name in names:
        v, f, _, _ = load_mesh(osp.join(root, name + ".obj"))
        models[name] = (v, f)
    _, _, uvcoords, uvfaces = load_mesh(osp.join("creadto-model", "template", "naked_body.obj"))
    
    return models, uvcoords, uvfaces

def save_texture_models(root, op_dict, uvcoord, uvface):
    from creadto.utils.io import save_mesh
    from torchvision.transforms.functional import to_pil_image
    
    if osp.exists(root) is False:
        os.mkdir(root)
    
    uvcoord = uvcoord.cpu().detach().numpy()
    uvface = uvface.cpu().detach().numpy()
    for name, vft in op_dict.items():
        vertex, face, texture = vft
        # to numpy and cv2
        vertex = vertex.cpu().detach().numpy()
        face = face.cpu().detach().numpy()
        save_mesh(osp.join(root, name + ".obj"), vertex, face,
                  texture=to_pil_image(texture / 255.), uvcoords=uvcoord, uvfaces=uvface)
        
def procedure(root, mini_batch_size=16):
    from creadto.utils.io import save_mesh
    from torchvision.transforms.functional import to_pil_image
    name_card = {
        'in-image': 'input-images',
        'in-plane': 'modeling-plane',
        'i-head': 'image-head-cropped',
        'i-head-albedo': 'image-head-albedo',
        'i-head-raw': 'image-head-albedo-raw',
        'i-naked-raw': 'image-naked-albedo-raw',
        'i-face-enhanced': 'image-face-enhanced',
        'i-face-parsing': 'image-face-parsing',
        'i-skin-filtered': 'image-skin-filtered',
        'i-albedo-mask': 'image-albedo-mask',
        'i-albedo-mask-d': 'image-albedo-mask-detail',
        'i-head-normal': 'image-head-normal',
        'p-head-landmark': 'parameter-face-landmark',
        't-naked': 'texture-naked',
    }
    for key, value in name_card.items():
        if 'in-' == key[:3]:
            continue
        if osp.exists(osp.join(root, value)):
            shutil.rmtree(osp.join(root, value))
        os.mkdir(osp.join(root, value))
        
    # must be pair a set of vertex and faces and an texture image
    tex_pack = image_to_texture(osp.join(root, name_card['in-image']), mini_batch_size)
    for mini_pack in tex_pack:
        tex_dict, names = mini_pack
        models, uvcoords, uvfaces = load_plane_models(osp.join(root, name_card['in-plane']), names)
        
        image_pack = {
            name_card['i-head']: tex_dict['head_images'],
            name_card['i-head-albedo']: tex_dict['head_albedos'],
            name_card['i-head-raw']: tex_dict['head_albedos_raw'],
            name_card['i-face-parsing']: tex_dict['segmented_images'],
            name_card['i-face-enhanced']: tex_dict['enhanced_images'],
            name_card['i-skin-filtered']: tex_dict['filtered_images'],
            name_card['i-albedo-mask']: tex_dict['uv_mask'],
            name_card['i-naked-raw']: tex_dict['full_albedos_raw'],
            name_card['i-head-normal']: tex_dict['head_normal_map']
        }

        for i, name in enumerate(names):
            v, f = models[name]
            models[name] = (v, f, tex_dict['full_albedos'][i], tex_dict['normal_map'][i])
            for key, value in image_pack.items():
                image = value[i]
                pil_image = torchvision.transforms.functional.to_pil_image(image)
                image_path = osp.join(root, key, name + ".png") 
                pil_image.save(image_path)

            landmarks2d = tex_dict['landmarks2d'][i].cpu().detach().numpy()
            np.save(osp.join(root, name_card['p-head-landmark'], name + ".npy"), landmarks2d)
        
        for key, value in tex_dict['face_masks'].items():
            folder_path = osp.join(root, name_card['i-albedo-mask-d'], key)
            if osp.exists(folder_path):
                shutil.rmtree(folder_path)
            os.mkdir(folder_path)
            for i, mask in enumerate(value):
                pil_image = torchvision.transforms.functional.to_pil_image(mask)
                image_path = osp.join(folder_path, names[i] + ".png") 
                pil_image.save(image_path)
        
        uvcoord = uvcoords.cpu().detach().numpy()
        uvface = uvfaces.cpu().detach().numpy()
        for name, vft in models.items():
            vertex, face, texture, normal_map = vft
            # to numpy and cv2
            vertex = vertex.cpu().detach().numpy()
            face = face.cpu().detach().numpy()
            
            naked_path = osp.join(root, name_card['t-naked'], name + ".obj")       
            save_mesh(naked_path, vertex, face, texture=to_pil_image(texture / 255.),
                    uvcoords=uvcoord, uvfaces=uvface, normal_map=to_pil_image(normal_map))
    