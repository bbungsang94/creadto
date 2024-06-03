from PIL import Image
from torchvision.transforms import ToTensor
from creadto.models.recon import DetailFaceModel


def run(path: str):
    model = DetailFaceModel()
    pil = Image.open(path)
    trans = ToTensor()
    image_tensor = trans(pil)

    import numpy as np
    from skimage.io import imread
    image = np.array(imread(path))

    result = model(trans(pil))
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    v = result['trans_verts']
    coordi = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.vertices = o3d.utility.Vector3dVector(v[0].numpy())
    mesh.triangles = o3d.utility.Vector3iVector(result['faces'])
    o3d.visualization.draw_geometries([coordi, mesh])
    pass
