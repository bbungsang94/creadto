import numpy as np
import open3d as o3d
from creadto.services.reconstruction import reconstruct_body_from_directory, reconstruct_body_from_file

def visualize_meshes(vertex, face, **kwargs):
    meshes = []
    for i in range(vertex.shape[0]):
        # Mesh 생성
        mesh = o3d.geometry.TriangleMesh()
        v = vertex[i].cpu().detach().numpy()
        mesh.vertices = o3d.utility.Vector3dVector(v + np.array([i * 0.5, 0, 0]))  # x축으로 0.5m 이동
        mesh.triangles = o3d.utility.Vector3iVector(face)
        mesh.compute_vertex_normals()  # 노멀 계산
        
        meshes.append(mesh)

    # Open3D로 시각화
    o3d.visualization.draw_geometries(meshes)
    
result = reconstruct_body_from_file("./sample/m-daniel-half.jpeg")
visualize_meshes(**result)

result = reconstruct_body_from_directory("./sample")
visualize_meshes(**result)