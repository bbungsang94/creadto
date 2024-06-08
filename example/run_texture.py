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
    return model(raw_images)

def load_plane_models(root):
    from creadto.utils.io import load_mesh
    files = os.listdir(root)
    models = []
    for i, file in enumerate(files):
        v, f, _, _ = load_mesh(osp.join(root, file))
        models.append((v, f))
    _, _, uvcoords, uvfaces = load_mesh(osp.join("creadto-model", "template", "naked_body.obj"))
    
    return models, uvcoords, uvfaces

def save_texture_models(root, names, vf_set, textures, uvcoord, uvface):
    from creadto.utils.io import save_mesh
    from torchvision.transforms.functional import to_pil_image
    
    if osp.exists(root) is False:
        os.mkdir(root)
    
    uvcoord = uvcoord.cpu().detach().numpy()
    uvface = uvface.cpu().detach().numpy()
    for vf, texture, name in zip(vf_set, textures, names):
        vertex, face = vf
        # to numpy and cv2
        vertex = vertex.cpu().detach().numpy()
        face = face.cpu().detach().numpy()
        save_mesh(osp.join(root, name + ".obj"), vertex, face,
                  texture=to_pil_image(texture), uvcoords=uvcoord, uvfaces=uvface)
        
def procedure(root):
    models, uvcoords, uvfaces = load_plane_models(osp.join(root, "plane_model"))
    full_texture = image_to_texture(osp.join(root, "input_images"))
    files = os.listdir(osp.join(root, "input_images"))
    files = [x.split('.')[0] for x in files]
    save_texture_models(osp.join(root, "texture-model"), files, models, full_texture, uvcoords, uvfaces)