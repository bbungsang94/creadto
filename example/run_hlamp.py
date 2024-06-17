import os
import shutil
import os.path as osp
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

    files = os.listdir(root)
    files = [os.path.join(root, x) for x in files]
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
    # files = os.listdir(os.path.join(root, "raw"))
    files = os.listdir(root)
    files = [os.path.join(root, x) for x in files]
    #files = [os.path.join(root, "raw", x) for x in files]
    raw_images = []
    h = 450
    w = 300
    for i, file in enumerate(files):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        raw_images.append(trans(image))
    raw_images = torch.stack(raw_images, dim=0)
    result = flaep(raw_images)
    result['names'] = files
    return result, flaep.reconstructor


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


def procedure(root):
    import trimesh
    import open3d as o3d
    from creadto.models.legacy import ModelConcatenator
    from creadto.utils.io import save_mesh
    concatenator = ModelConcatenator(root="./creadto-model/template")
    face_model, recon_model = image_to_flaep(root=osp.join(root, "input_images"))
    gender = image_to_gender(images=face_model['crop_image'])
    body_model = image_to_blass(root=osp.join(root, "input_images"))
    body_measurement = body_to_measure(body_model['plane_vertex'], gender)
    face_measurement = head_to_measure(face_model['plane_verts'])
    humans = concatenator.update_model(body=body_model['plane_vertex'], head=face_model['plane_verts'], visualize=False)
    
    if osp.exists(osp.join(root, "posed_model")):
        shutil.rmtree(osp.join(root, "posed_model"))
    os.mkdir(osp.join(root, "posed_model"))
    if osp.exists(osp.join(root, "plane_model")):
        shutil.rmtree(osp.join(root, "plane_model"))
    os.mkdir(osp.join(root, "plane_model"))
    if osp.exists(osp.join(root, "head_model")):
        shutil.rmtree(osp.join(root, "head_model"))
    os.mkdir(osp.join(root, "head_model"))
    if osp.exists(osp.join(root, "pose_parameter")):
        shutil.rmtree(osp.join(root, "pose_parameter"))
    os.mkdir(osp.join(root, "pose_parameter"))
    
    for i, pose in enumerate(body_model['shape_parameters']['pose']):
        filename = face_model['names'][i].split('.')[0] + ".pth"
        filename = filename.replace("input_images", "pose_parameter")
        torch.save(pose, filename)
        
    # save only vertices and faces, no textures, no normal maps, no materials
    for i, tup in enumerate(zip(body_model['vertex'], humans['model']['body'])):
        v, pv = tup
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pv.cpu().detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(body_model['face'])
        # o3d.visualization.draw_geometries([mesh])
        file_path = face_model['names'][i].split('.')[0]
        pose_path = file_path.replace("input_images", "posed_model")
        save_mesh(obj_name=pose_path + ".obj", vertices=v, faces=body_model['face'])
        plane_path = file_path.replace("input_images", "plane_model")
        save_mesh(obj_name=plane_path + ".obj", vertices=pv, faces=body_model['face'])
        head_path = file_path.replace("input_images", "head_model")
        head_face = face_model['faces'].cpu().detach().numpy()
        coarse_head_vertex = humans['model']['coarse_head'][i]
        save_mesh(obj_name=head_path + "-ch.obj", vertices=coarse_head_vertex, faces=head_face)
        head_vertex = humans['model']['head'][i]
        save_mesh(obj_name=head_path + "-h.obj", vertices=head_vertex, faces=head_face)
    # for i, v in enumerate(humans['model']['body']):
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(v.cpu().detach().numpy())
    #     mesh.triangles = o3d.utility.Vector3iVector(body_model['face'])
    #     file_path = face_model['names'][i].split('.')[0]
    #     file_path = file_path.replace("input_images", "plane_model")
    #     o3d.io.write_triangle_mesh(file_path + ".obj", mesh)