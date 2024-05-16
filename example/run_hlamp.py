import os
import cv2
import torch
import torchvision
import numpy as np


def image_to_gender(images):
    from creadto.models.det import GenderClassification
    classifier = GenderClassification()

    return classifier(images)


def image_to_blass(root):
    from creadto.models.recon import BLASS

    files = os.listdir(os.path.join(root, "raw"))
    files = [os.path.join(root, "raw", x) for x in files]
    raw_images = []
    hlamp = BLASS()
    h = 450
    w = 300
    for i, file in enumerate(files):
        image = cv2.imread(file)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        raw_images.append(image)
    raw_images = np.stack(raw_images, axis=0)
    return hlamp(raw_images)


def image_to_flaep(root):
    from creadto.models.recon import DetailFaceModel
    trans = torchvision.transforms.ToTensor()
    flaep = DetailFaceModel()
    files = os.listdir(os.path.join(root, "raw"))
    files = [os.path.join(root, "raw", x) for x in files]
    raw_images = []
    h = 450
    w = 300
    for i, file in enumerate(files):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        raw_images.append(trans(image))
    raw_images = torch.stack(raw_images, dim=0)
    return flaep(raw_images)


def body_to_measure(vertices, gender):
    from creadto.models.legacy import GraphTailor
    from creadto._external.smpl.smpl import SMPL

    dim_guide = torch.load(os.path.join('./creadto-model', 'body_dimension_guide.pt'))
    tailor = GraphTailor(dim_guide)
    smpl = SMPL("./creadto-model")

    bodies = dict()
    for pose, value in dim_guide['poses'].items():
        pose_param = value.type(torch.float32)
        pose_param = pose_param.expand(vertices.shape[0], pose_param.shape[1], pose_param.shape[2])
        v = smpl.pose_to(vertices, pose_param.detach().clone())

        bodies[pose] = v.detach().clone()

    tailor.update(bodies)
    return tailor.order(gender=gender, visualize=False, normalize=False)


def head_to_measure(vertices):
    from creadto.models.legacy import GraphTailor
    from creadto._external.flame.flame import FLAME

    dim_guide = torch.load(os.path.join('./creadto-model', 'head_dimension_guide.pt'))
    tailor = GraphTailor(dim_guide)
    smpl = FLAME("./creadto-model")

    bodies = {'standard': vertices.detach().clone()}

    tailor.update(bodies)
    return tailor.order(gender=["female"] * vertices.shape[0], visualize=False, normalize=False)


def procedure():
    import open3d as o3d
    from creadto.models.legacy import ModelConcatenator
    concatenator = ModelConcatenator(root="./creadto-model")
    face_model = image_to_flaep(root=r"D:\dump")
    gender = image_to_gender(images=face_model['crop_image'])
    body_model = image_to_blass(root=r"D:\dump")

    for i, v in enumerate(body_model['vertex']):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v.cpu().detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(body_model['face'])
        o3d.io.write_triangle_mesh(os.path.join("D:\dump\model", "%05d.obj" % i), mesh)
    body_measurement = body_to_measure(body_model['plane_vertex'], gender)
    face_measurement = head_to_measure(face_model['trans_verts'])
    humans = concatenator.update_model(body=body_model['plane_vertex'], head=face_model['plane_verts'], visualize=True)
    for i, v in enumerate(humans['model']['body']):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v.cpu().detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(body_model['face'])
        o3d.io.write_triangle_mesh(os.path.join("D:\dump\plane", "%05d.obj" % i), mesh)
