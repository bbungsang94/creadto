import os
import shutil
import os.path as osp
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image


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
    for i, file in enumerate(files):
        image = Image.open(osp.join(root, file))
        raw_images.append(image)
    return hlamp(raw_images)


def image_to_flaep(root):
    from creadto.models.recon import DetailFaceModel
    flaep = DetailFaceModel()
    # files = os.listdir(os.path.join(root, "raw"))
    files = os.listdir(root)
    files = [os.path.join(root, x) for x in files]
    #files = [os.path.join(root, "raw", x) for x in files]
    raw_images = []
    for i, file in enumerate(files):
        image = Image.open(osp.join(root, file))
        raw_images.append(image)
    crop_images, process = flaep.encode_pil(raw_images)
    result = flaep.decode(crop_images)
    result['process'] = process
    result['names'] = files
    return result


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
    
    name_card = {
        'in-image': 'input-images',
        'm-posed': 'modeling-posed',
        'm-plane': 'modeling-plane',
        'm-head': 'modeling-head',
        'm-densehead': 'modeling-densehead',
        'u-measurement': 'utility-measurement',
        'p-body-pose': 'parameter-body-pose',
        'i-normal-head': 'image-normal-head',
    }
    for key, value in name_card.items():
        if 'in-' == key[:3]:
            continue
        if osp.exists(osp.join(root, value)):
            shutil.rmtree(osp.join(root, value))
        os.mkdir(osp.join(root, value))
        
    concatenator = ModelConcatenator(root="./creadto-model/template")
    face_model = image_to_flaep(root=osp.join(root, name_card['in-image']))
    gender = image_to_gender(images=face_model['crop_image'])
    body_model = image_to_blass(root=osp.join(root, name_card['in-image']))
    body_measurement = body_to_measure(body_model['plane_vertex'], gender)
    face_measurement = head_to_measure(face_model['plane_verts'])
    humans = concatenator.update_model(body=body_model['plane_vertex'], head=face_model['plane_verts'], visualize=False)
    
    for i, pose in enumerate(body_model['shape_parameters']['pose']):
        filename = face_model['names'][i].split('.')[0] + ".pth"
        filename = filename.replace(name_card['in-image'], name_card['p-body-pose'])
        torch.save(pose, filename)
    
    pack = zip(face_model['dense_verts'], face_model['dense_faces'], face_model['dense_colors'])
    for i, tup in enumerate(pack):
        v, f, c = tup
        file_path = face_model['names'][i].split('.')[0]
        detail_path = file_path.replace(name_card['in-image'], name_card['m-densehead'])
        save_mesh(obj_name=detail_path + ".obj", vertices=v, faces=f, colors=c)
        
        dnor = face_model['uv_detail_normals'][i]
        dnor_pil = torchvision.transforms.functional.to_pil_image(dnor)
        normal_path = file_path.replace(name_card['in-image'], name_card['i-normal-head'])
        dnor_pil.save(normal_path + ".png")
                
    # save only vertices and faces, no textures, no normal maps, no materials
    for i, tup in enumerate(zip(body_model['vertex'], humans['model']['body'])):
        v, pv = tup
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pv.cpu().detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(body_model['face'])
        # o3d.visualization.draw_geometries([mesh])
        file_path = face_model['names'][i].split('.')[0]
        pose_path = file_path.replace(name_card['in-image'], name_card['m-posed'])
        save_mesh(obj_name=pose_path + ".obj", vertices=v, faces=body_model['face'])
        plane_path = file_path.replace(name_card['in-image'], name_card['m-plane'])
        save_mesh(obj_name=plane_path + ".obj", vertices=pv, faces=body_model['face'])
        head_path = file_path.replace(name_card['in-image'], name_card['m-head'])
        head_face = face_model['faces'].cpu().detach().numpy()
        coarse_head_vertex = humans['model']['coarse_head'][i]
        save_mesh(obj_name=head_path + "-ch.obj", vertices=coarse_head_vertex, faces=head_face)
        head_vertex = humans['model']['head'][i]
        save_mesh(obj_name=head_path + "-h.obj", vertices=head_vertex, faces=head_face)
