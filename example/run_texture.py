import os
import os.path as osp
import trimesh
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
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        models.append(mesh)
    return models
    
def procedure(root):
    models = load_plane_models(osp.join(root, "plane_model"))
    full_texture = image_to_texture(osp.join(root, "input_images"))