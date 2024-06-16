import os
import os.path as osp
from PIL import Image


def image_to_texture(root):
    from creadto.models.tex import NakedHuman
    
    model = NakedHuman()
    files = os.listdir(root)
    raw_images = []
    for i, file in enumerate(files):
        image = Image.open(osp.join(root, file))
        raw_images.append(image)
    return model(raw_images), files

def load_plane_models(root):
    from creadto.utils.io import load_mesh
    files = os.listdir(root)
    models = dict()
    for i, file in enumerate(files):
        v, f, _, _ = load_mesh(osp.join(root, file))
        models[file.split('.')[0]] = (v, f)
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
                  texture=to_pil_image(texture), uvcoords=uvcoord, uvfaces=uvface)
        
def procedure(root):
    import torchvision
    # must be pair a set of vertex and faces and an texture image
    models, uvcoords, uvfaces = load_plane_models(osp.join(root, "plane_model"))
    result_dict, names = image_to_texture(osp.join(root, "input_images"))
    
    if osp.exists(osp.join(root, "head-texture")) is False:
        os.mkdir(osp.join(root, "head-texture"))
    for i, head_albedo in enumerate(result_dict['head_texture']):
        pil_image = torchvision.transforms.functional.to_pil_image(head_albedo.cpu().detach())
        pil_image.save(osp.join(root, "head-texture", "%d-th head_texture.png" % i))
    for i, name in enumerate(names):
        name = name.split('.')[0]
        v, f = models[name]
        models[name] = (v, f, result_dict['full_texture'][i])
        
    save_texture_models(osp.join(root, "texture-model"), models, uvcoords, uvfaces)