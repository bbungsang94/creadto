import os
import time
import cv2
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.io import savemat
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from creadto._external.deca.decalib.datasets import datasets
from creadto._external.deca.decalib.deca import DECA
from creadto._external.deca.decalib.utils import util
from creadto._external.deca.decalib.deca import cfg as deca_cfg


def get_facial_model(full_path="../_external/mediapipe/face_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=full_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=False,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def main(path=r"D:\Creadto\Heritage\Dataset\ffhq-dataset\images1024x1024\00002.png"):
    trans = ToPILImage()
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = "./results"
    device = "cuda"
    os.makedirs(savefolder, exist_ok=True)

    # load test images
    testdata = datasets.TestData(path, iscrop=True, face_detector="fan", sample_step=10)

    # run DECA
    # deca_cfg.model.use_tex = True
    # deca_cfg.rasterizer_type = args.rasterizer_type
    # deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)  # tensor
            # Render results in original image size
            tform = testdata[i]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1, 2).to(device)
            original_image = testdata[i]['original_image'][None, ...].to(device)
            _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
            orig_visdict['inputs'] = original_image

        os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        # save depth
        depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
        visdict['depth_images'] = depth_image
        cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        # save keypoints
        np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
        np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        # save objects
        deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        # save material
        opdict = util.dict_tensor2npy(opdict)
        savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        # save visualization output
        cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
        cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
        # save image information
        for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images',
                         'landmarks2d']:
            if vis_name not in visdict.keys():
                continue
            image = util.tensor2image(visdict[vis_name][0])
            cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
                        util.tensor2image(visdict[vis_name][0]))

            image = util.tensor2image(orig_visdict[vis_name][0])
            cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name + '.jpg'),
                        util.tensor2image(orig_visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')


def infer_model(deca, images):
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict)  # tensor

    opdict['shape'] = codedict['shape']
    return opdict, visdict


def eval_face_set(root=r"D:\dump"):
    import pandas as pd
    import open3d as o3d
    import random
    results = {'category': [], 'class': [], 'sample': [], 'process_time': []}
    values = []

    device = "cuda"
    deca = DECA(config=deca_cfg, device=device)
    folders = os.listdir(root)
    random.shuffle(folders)
    # old_set = pd.read_csv('eval_head_result.csv', index_col=0)
    for folder in tqdm(folders):
        folder_path = os.path.join(root, folder)
        files = os.listdir(folder_path)
        resume_index = 0
        # for i, file in enumerate(files):
        #     t = old_set[old_set['sample'] == file.replace('.png', '')]
        #     if len(t) > 0:
        #         resume_index = i + 1
        #         continue
        #     else:
        #         resume_index = i
        #         break
        # if resume_index == len(files):
        #     continue
        testdata = datasets.TestData(folder_path, iscrop=True, face_detector="fan", sample_step=10)
        for i in range(resume_index, len(testdata)):
            start = time.time()
            images = testdata[i]['image'].to(device)[None, ...]
            op, vis = infer_model(deca, images)
            # if i == 0:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(op['trans_verts'][0].cpu().detach().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(deca.flame.faces_tensor.cpu().numpy())
            #o3d.visualization.draw_geometries([mesh])
            # save objects
            deca.save_obj(os.path.join(root.replace('images', '3dmodels'), folder, testdata[i]['imagename'] + '.obj'), op)
            # save material
            opdict = util.dict_tensor2npy(op)
            savemat(os.path.join(root.replace('images', '3dmodels'), folder, testdata[i]['imagename'] + '.mat'), opdict)

            shape = op['shape'][0].cpu().detach().numpy()
            results['category'].append('head')
            results['class'].append(folder)
            results['sample'].append(testdata[i]['imagename'])
            shape = (shape + 2) / 4
            values.append(shape)
            end = time.time()
            results['process_time'].append(end - start)
        info = pd.DataFrame(results)
        contents = pd.DataFrame(values)
        #eval_set = info.join(contents)
        #eval_set = pd.concat([old_set, eval_set], ignore_index=True)
        #eval_set.to_csv('eval_head_result.csv')
            # import open3d as o3d
            # pcd_verts = o3d.geometry.PointCloud()
            # pcd_verts.points = o3d.utility.Vector3dVector(op['verts'][0].cpu().detach().numpy())
            #
            # pcd_trans_verts = o3d.geometry.PointCloud()
            # pcd_trans_verts.points = o3d.utility.Vector3dVector(op['trans_verts'][0].cpu().detach().numpy())
            # pcd_trans_verts.paint_uniform_color((1.0, 0, 0))
            #
            # pcd_landmark = o3d.geometry.PointCloud()
            # pcd_landmark.points = o3d.utility.Vector3dVector(op['landmarks3d'][0, :, :3].cpu().detach().numpy())
            #
            # coordi = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # o3d.visualization.draw_geometries([pcd_landmark, coordi])
            # pass
            # Render results in original image size
            # tform = testdata[i]['tform'][None, ...]
            # tform = torch.inverse(tform).transpose(1, 2).to(device)
            # original_image = testdata[i]['original_image'][None, ...].to(device)
            # _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
            # orig_visdict['inputs'] = original_image

if __name__ == "__main__":
    eval_face_set()
