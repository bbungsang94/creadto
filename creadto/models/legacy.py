import copy
import os
import pickle

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data, Batch


class ModelConcatenator:
    def __init__(self, root):
        with open(file=os.path.join(root, "MANO_SMPLX_vertex_ids.pkl"), mode='rb') as f:
            self.ids = pickle.load(f)
        self.ids.update({'head': np.load(os.path.join(root, "SMPL-X__FLAME_vertex_ids.npy"))})
        self.ids.update({'neck': np.load(os.path.join(root, "body_neck_indices.npy"))})
        self.human = {
            'model': {
                'body': None,
                'head': None,
                'face': None
            },
            'rig': {

            }
        }

    def update_model(self, **kwargs):
        vertex = self.human['model']
        for k, v in kwargs.items():
            if k in vertex:
                vertex[k] = v

        if vertex['body'] is not None:
            # fitting models into body
            # 1. Scale dismatched, corn head and muscle of back
            body = vertex['body']
            if vertex['head'] is not None:
                head = vertex['head']
                partial_neck = body[:, self.ids['neck'], :]
                inplace_max, _ = partial_neck.max(dim=1)
                inplace_min, _ = partial_neck.min(dim=1)
                
                body[:, self.ids['head'], :] = head
                target_neck = body[:, self.ids['neck'], :]
                target_max, _ = target_neck.max(dim=1)
                target_min, _ = target_neck.min(dim=1)
                
                ratio = (target_max - target_min) / (inplace_max - inplace_min)
                body[:, self.ids['head'], :] = head * ratio.unsqueeze(dim=1)
                target_neck = body[:, self.ids['neck'], :]
                target_min, _ = target_neck.min(dim=1)

                pivot = inplace_min - target_min
                
                body[:, self.ids['head'], :] = body[:, self.ids['head'], :] + pivot.unsqueeze(dim=1)
                self.human['model']['body'] = body
                self.human['model']['head'] = copy.deepcopy(body[:, self.ids['head'], :])
        if 'visualize' in kwargs and kwargs['visualize']:
            import open3d as o3d
            mesh = o3d.geometry.PointCloud()
            mesh.points = o3d.utility.Vector3dVector(self.human['model']['body'][0].detach().numpy())
            o3d.visualization.draw_geometries([mesh])
        return self.human


class Tailor:
    def __init__(self, tape, pin, circ_dict, norm_range, model_dict=None):
        """
        :param tape: It's interaction list from convention.py. call get_interactions
        :param pin: Vertices index about human parts in 3d mesh model load standing/sitting.json
        """

        self.tape = tape
        self.pins = self._convert_tag(pin)
        self.pins = self._delete_island()
        self.table = torch.zeros(1, len(self.tape))
        self.model = model_dict
        self.circ_dict = circ_dict
        self.norm_range = norm_range

    def update(self, model_dict):
        """
        :param model_dict: vertices ("standing", "sitting", "t", "hands-on", "curve")
        :return:
        """
        self.model = model_dict
        self.table = torch.zeros(self.model[list(model_dict.keys())[-1]].shape[0], len(self.tape))

    def dump(self, gender, fast=False, normalize=False, visualize=False):
        only_female = ["Waist Height natural indentation", "Underbust Circumference"]
        only_female_idx = []
        gender = np.array(gender)
        vis_idx = 0
        for i, paper in enumerate(self.tape):
            kor, eng, tags, func, pose = paper
            vertex = copy.deepcopy(self.model[pose])
            # points name to index
            args = []
            for tag in tags:
                index = self.pins[tag]
                point = vertex[:, index].detach().cpu()
                args.append(point)

            # function 인식
            if fast and ("circ" in func or "length" in func):
                continue
            if "circ" in func or 'length' in func:
                indexes = self.circ_dict[eng]
                args.clear()
                for index in indexes:
                    args.append(vertex[:, index].detach().cpu())

                if '-' in func:
                    stub = func.split('-')
                    args.insert(0, stub[-1])
                else:
                    stub = [func]
                    args.insert(0, 'straight')
                func = stub[0]

            value = getattr(self, func)(*args)
            self.table[:, i] = value
            if "귀길이" in kor and value.min() < 0.011:
                vis_idx = value.argmin().item()
            if eng in only_female:
                only_female_idx.append(i)

        # cut out-range
        table = self.table.numpy()
        if self.norm_range['max'].shape[0] != self.table.shape[1]:
            merge_idx = self.norm_range['split_index']
            temp = np.zeros((self.table.shape[0], self.norm_range['max'].shape[0]), dtype=np.float32)
            for i in range(self.table.shape[1]):
                temp[:, merge_idx[i]] += table[:, i]
            table = temp

        valid_indices = np.all((table <= self.norm_range['max']) & (table >= self.norm_range['min']), axis=1)
        self.table = self.table[valid_indices]
        male_indexes = np.where(gender[valid_indices] == "male")[0]
        if normalize:
            self.table = self.normalize(self.table)

        if len(only_female_idx) != 0:
            self.table[male_indexes, only_female_idx[0]] = 0.
            self.table[male_indexes, only_female_idx[1]] = 0.
        if visualize:
            vis_idx = 23
            print(vis_idx)
            for pose, vertex in self.model.items():
                points = []
                lines = []
                line_colors = []
                v = vertex[vis_idx].numpy()
                colors = np.zeros((v.shape[0], 3))
                colors += 0.4
                for paper in self.tape:
                    kor, eng, tags, func, p = paper
                    if p != pose:
                        continue
                    if "circ" in func:
                        indexes = self.circ_dict[eng]
                    elif "length" in func and eng in self.circ_dict:
                        indexes = self.circ_dict[eng]
                    else:
                        indexes = [self.pins[tag] for tag in tags]
                    for i, index in enumerate(indexes):
                        point = copy.deepcopy(v[index])
                        if "width" in func:
                            point[1] = v[indexes[0]][1]
                            point[2] = v[indexes[0]][2]
                            line_colors.append([1., 0., 0.])
                        elif "depth" in func:
                            point[0] = v[indexes[0]][0]
                            point[1] = v[indexes[0]][1]
                            line_colors.append([0., 1., 0.])
                        elif "height" in func:
                            point[0] = v[indexes[0]][0]
                            point[2] = v[indexes[0]][2]
                            line_colors.append([0., 0., 1.])
                        else:
                            line_colors.append([1., 0., 1.])
                        points.append(point)
                        lines.append([i, ((i + 1) % len(indexes))])

                    if "length" in func:
                        del lines[-1]
                        del line_colors[-1]

                    print(kor)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(v)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector(np.array(points))
                    line.lines = o3d.utility.Vector2iVector(np.array(lines))
                    line.colors = o3d.utility.Vector3dVector(np.array(line_colors))
                    o3d.visualization.draw_geometries([line, pcd])
                    points.clear()
                    lines.clear()
                    line_colors.clear()

        return self.table, valid_indices

    def order(self, gender, fast=False, normalize=False, visualize=False):
        only_female = ["Waist Height natural indentation", "Underbust Circumference"]
        only_female_idx = []
        gender = np.array(gender)
        vis_idx = 0
        for i, paper in enumerate(self.tape):
            kor, eng, tags, func, pose = paper
            vertex = copy.deepcopy(self.model[pose])
            # points name to index
            args = []
            for tag in tags:
                index = self.pins[tag]
                point = vertex[:, index].detach().cpu()
                args.append(point)

            # function 인식
            if fast and ("circ" in func or "length" in func):
                continue
            if "circ" in func or 'length' in func:
                indexes = self.circ_dict[eng]
                args.clear()
                for index in indexes:
                    args.append(vertex[:, index].detach().cpu())

                if '-' in func:
                    stub = func.split('-')
                    args.insert(0, stub[-1])
                else:
                    stub = [func]
                    args.insert(0, 'straight')
                func = stub[0]

            value = getattr(self, func)(*args)
            self.table[:, i] = value
            if eng in only_female:
                only_female_idx.append(i)

        male_indexes = np.where(gender == "male")[0]
        if normalize:
            self.table = self.normalize(self.table)

        if len(only_female_idx) != 0:
            self.table[male_indexes, only_female_idx[0]] = 0.
            self.table[male_indexes, only_female_idx[1]] = 0.
        if visualize:
            vis_idx = 0
            print(vis_idx)
            for pose, vertex in self.model.items():
                points = []
                lines = []
                line_colors = []
                v = vertex[vis_idx].numpy()
                colors = np.zeros((v.shape[0], 3))
                colors += 0.4
                for paper in self.tape:
                    kor, eng, tags, func, p = paper
                    if p != pose:
                        continue
                    if "circ" in func:
                        indexes = self.circ_dict[eng]
                    elif "length" in func and eng in self.circ_dict:
                        indexes = self.circ_dict[eng]
                    else:
                        indexes = [self.pins[tag] for tag in tags]
                    for i, index in enumerate(indexes):
                        point = copy.deepcopy(v[index])
                        if "width" in func:
                            point[1] = v[indexes[0]][1]
                            point[2] = v[indexes[0]][2]
                            line_colors.append([1., 0., 0.])
                        elif "depth" in func:
                            point[0] = v[indexes[0]][0]
                            point[1] = v[indexes[0]][1]
                            line_colors.append([0., 1., 0.])
                        elif "height" in func:
                            point[0] = v[indexes[0]][0]
                            point[2] = v[indexes[0]][2]
                            line_colors.append([0., 0., 1.])
                        else:
                            line_colors.append([1., 0., 1.])
                        points.append(point)
                        lines.append([i, ((i + 1) % len(indexes))])

                    if "length" in func:
                        del lines[-1]
                        del line_colors[-1]

                    print(kor)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(v)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector(np.array(points))
                    line.lines = o3d.utility.Vector2iVector(np.array(lines))
                    line.colors = o3d.utility.Vector3dVector(np.array(line_colors))
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    vis.add_geometry(pcd)
                    vis.add_geometry(line)
                    vis.get_render_option().line_width = 10
                    vis.get_render_option().point_size = 2
                    vis.run()
                    points.clear()
                    lines.clear()
                    line_colors.clear()

        if self.table.shape[1] >= 46:
            self.table[:, 39] = self.table[:, 39] - 0.0791707
            self.table[:, 32] = self.table[:, 32] - 0.063667944
            self.table[:, 28] = self.table[:, 28] - 0.058014839
            self.table[:, 3] = self.table[:, 3] - 0.055395142
            self.table[:, 6] = self.table[:, 6] - 0.048676284
            self.table[:, 20] = self.table[:, 20] - 0.038910605
            self.table[:, 36] = self.table[:, 36] + 0.04487607
            self.table[:, 46] = self.table[:, 46] + 0.031787722
            self.table[:, 44] = self.table[:, 44] - 0.035034574
            self.table[:, 42] = self.table[:, 42] + 0.015055184
            self.table[:, 41] = self.table[:,41] - 0.028909176
            self.table[:, 40] = self.table[:, 40] - 0.016079659
            self.table[:, 25] = self.table[:, 25] + 0.011062523
            self.table[:, 22] = self.table[:, 22] + 0.023733191
            self.table[:, 16] = self.table[:, 16] - 0.02816479
            self.table[:, 15] = self.table[:, 15] + 0.006243895
            self.table[:, 4] = self.table[:, 4] - 0.032512034
            self.table[:, 11] = self.table[:, 11] + 0.011548213
            self.table[:, 13] = self.table[:, 13] - 0.01753491
            self.table[:, 7] = self.table[:, 7] - 0.020710816
            self.table[:, 27] = self.table[:, 27] - 0.014126228
            self.table[:, 30] = self.table[:, 30] - 0.026458427
        return self.table

    def _convert_tag(self, pin):
        tag = dict()
        for pose in pin.keys():
            dictionary = pin[pose]
            for key, value in dictionary.items():
                _, eng, direction = self._separate_key(key)
                tag[eng + direction] = value
        return tag

    def _delete_island(self):
        new_pins = dict()
        for tape in self.tape:
            _, _, pins, _, _ = tape
            for pin in pins:
                if pin in self.pins:
                    new_pins[pin] = self.pins[pin]
        return new_pins

    def normalize(self, measure, scale=1):
        max_val = np.expand_dims(self.norm_range['max'], axis=0) * scale
        min_val = np.expand_dims(self.norm_range['min'], axis=0) * scale
        if self.norm_range['max'].shape[0] != measure.shape[1]:
            temp_max = np.zeros((max_val.shape[0], measure.shape[1]), dtype=np.float32)
            temp_min = np.zeros((min_val.shape[0], measure.shape[1]), dtype=np.float32)
            merge_idx = self.norm_range['split_index']
            weight = self.norm_range['split_weight']
            for i in range(measure.shape[1]):
                temp_max[:, i] = max_val[:, merge_idx[i]] * weight[i]
                temp_min[:, i] = min_val[:, merge_idx[i]] * weight[i]
            max_val = temp_max
            min_val = temp_min

        normalized = 2 * (measure - min_val) / (max_val - min_val) - 1
        return normalized

    def denormalize(self, normalized, scale=1):
        normalized = normalized.cpu().detach().numpy()
        max_val = np.expand_dims(self.norm_range['max'], axis=0) * scale
        min_val = np.expand_dims(self.norm_range['min'], axis=0) * scale
        normalized = (normalized + 1) / 2
        if self.norm_range['max'].shape[0] != normalized.shape[1]:
            temp_max = np.zeros((normalized.shape[1]), dtype=np.float32)
            temp_min = np.zeros((normalized.shape[1]), dtype=np.float32)
            merge_idx = self.norm_range['split_index']
            weight = self.norm_range['split_weight']
            for i in range(normalized.shape[1]):
                temp_max[i] = max_val[:, merge_idx[i]] * weight[i]
                temp_min[i] = min_val[:, merge_idx[i]] * weight[i]
            max_val = temp_max
            min_val = temp_min

            temp = np.zeros((normalized.shape[0], self.norm_range['max'].shape[0]), dtype=np.float32)
            for i in range(normalized.shape[1]):
                temp[:, merge_idx[i]] += normalized[:, i] * (max_val[i] - min_val[i]) + min_val[i]
            denormalized = temp
        else:
            denormalized = normalized * (max_val - min_val) + min_val
        return torch.tensor(denormalized, dtype=torch.float32)

    @staticmethod
    def _separate_key(name):
        stub = name.split(', ')
        if len(stub) == 2:
            direction = ""
        else:
            direction = ", " + stub[-1]
        return stub[0], stub[1], direction

    @staticmethod
    def width(a, b):
        return abs(a[:, 0] - b[:, 0])

    @staticmethod
    def height(a, b):
        return abs(a[:, 1] - b[:, 1])

    @staticmethod
    def depth(a, b):
        return abs(a[:, 2] - b[:, 2])

    def circ(self, direction, *args):
        result = self.length(direction, *args)

        pivot = {
            'v': 0,
            'h': 1,
        }
        if direction in pivot:
            fix = args[0][:, pivot[direction]]
        else:
            fix = None

        a = args[-1]
        b = args[0]
        if direction in pivot:
            if fix is not None:
                a[:, pivot[direction]] = fix
                b[:, pivot[direction]] = fix
        result += torch.linalg.vector_norm(b - a, dim=1)
        return result

    @staticmethod
    def length(direction, *args):
        pivot = {
            'v': 0,
            'h': 1,
        }
        if direction in pivot:
            fix = args[0][:, pivot[direction]]
        else:
            fix = None

        length = len(args)
        result = torch.zeros(args[0].shape[0])
        for i in range(length):
            a = args[i]
            if (i + 1) == length:
                break
            b = args[i + 1]
            if direction in pivot:
                if fix is not None:
                    a[:, pivot[direction]] = fix
                    b[:, pivot[direction]] = fix
            result += torch.linalg.vector_norm(b - a, dim=1)
        return result


class GraphTailor(Tailor):
    def __init__(self, guide):
        tape = guide['interactions']
        pin = guide['landmarks']
        circ_dict = guide['circumference dict']
        norm_range = guide["range"]

        super().__init__(tape, pin, circ_dict, norm_range)
        named_nodes = {}
        for i, key in enumerate(self.pins.keys()):
            named_nodes[key] = i
        edge_indexes = []
        for i, paper in enumerate(self.tape):
            kor, eng, tags, func, pose = paper
            if len(tags) > 2:
                raise "here!"
            edge_indexes.append([named_nodes[tags[0]], named_nodes[tags[-1]]])
            edge_indexes.append([named_nodes[tags[-1]], named_nodes[tags[0]]])

        self.named_nodes = named_nodes
        self.edge_indexes = torch.tensor(edge_indexes, dtype=torch.long).t()

        self.measure_code = {'width': -2, 'height': -1, 'depth': 0, 'length': 1, 'circ': 2,
                             'length-h': 1, 'length-v': 1, 'circ-h': 2, 'circ-v': 2}
        self.axis_code = {'width': -1, 'height': 0, 'depth': 1, 'length': 2, 'circ': 2,
                          'length-h': -1, 'length-v': 0, 'circ-h': -1, 'circ-v': 0}

    def get_measured_graph(self, gender, normalize=False, fast=False, visualize=False):
        batch_size = len(gender)
        measure = self.order(gender=gender, fast=fast, visualize=visualize)
        if normalize:
            min_vals, _ = torch.min(measure, dim=1, keepdim=True)
            max_vals, _ = torch.max(measure, dim=1, keepdim=True)
            measure = -1 + 2 * (measure - min_vals) / (max_vals - min_vals)
        named_poses = {}
        for i, key in enumerate(self.model.keys()):
            named_poses[key] = i

        graphs = []
        for i in range(batch_size):
            edge_features = np.zeros((len(self.tape) * 2, len(self.model) + 2))
            node_features = np.zeros((len(self.pins), len(self.model) * 3))
            # node
            for key, vertices in self.model.items():
                v = vertices[i]
                for node_name, index in self.pins.items():
                    x, y, z = v[index]
                    pivot = named_poses[key] * 3
                    node_features[self.named_nodes[node_name], pivot] = x.item()
                    node_features[self.named_nodes[node_name], pivot + 1] = y.item()
                    node_features[self.named_nodes[node_name], pivot + 2] = z.item()

                # edge feature
                for pivot, paper in enumerate(self.tape):
                    kor, eng, tags, func, pose = paper
                    edge_features[pivot * 2, named_poses[key]] = measure[i, pivot].item()
                    edge_features[pivot * 2, -1] = self.axis_code[func]
                    edge_features[pivot * 2, -2] = self.measure_code[func]

                    edge_features[pivot * 2 + 1, named_poses[key]] = measure[i, pivot].item()
                    edge_features[pivot * 2 + 1, -1] = self.axis_code[func]
                    edge_features[pivot * 2 + 1, -2] = self.measure_code[func]
            node_features = torch.tensor(node_features, dtype=torch.float32)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
            graph = Data(x=node_features.detach(),
                         edge_index=self.edge_indexes.detach().contiguous(),
                         edge_attr=edge_features.detach())
            if visualize:
                import networkx as nx
                import matplotlib.pyplot as plt

                # NetworkX 그래프로 변환
                G = nx.Graph()
                G.add_nodes_from(range(len(self.named_nodes)))
                G.add_edges_from(self.edge_indexes.t().tolist())

                # 노드 및 엣지 특성 추가
                for k in range(len(self.named_nodes)):
                    G.nodes[k]['features'] = node_features[k].numpy()

                for k, (src, tgt) in enumerate(self.edge_indexes.t().tolist()):
                    G[src][tgt]['features'] = edge_features[k].numpy()

                # 그래프 시각화
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=200, node_color='skyblue',
                        font_color='black', font_size=8)

                # 노드 및 엣지 특성 표시
                node_labels = nx.get_node_attributes(G, 'features')
                edge_labels = nx.get_edge_attributes(G, 'features')
                nx.draw_networkx_labels(G, pos, labels=node_labels)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                plt.show()
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        return measure, batch

    def to_graph(self, measure, poses: list, normalize=True, visualize=False):
        if normalize:
            min_vals, _ = torch.min(measure, dim=1, keepdim=True)
            max_vals, _ = torch.max(measure, dim=1, keepdim=True)
            measure = -1 + 2 * (measure - min_vals) / (max_vals - min_vals)
        named_poses = {}
        for i, key in enumerate(poses):
            named_poses[key] = i

        graphs = []
        for i in range(len(measure)):
            edge_features = np.zeros((len(self.tape) * 2, len(poses) + 2))
            node_features = np.zeros((len(self.pins), len(poses) * 3))
            # node
            for key, vertices in self.model.items():
                v = vertices[i].detach().numpy()
                for node_name, index in self.pins.items():
                    x, y, z = v[index]
                    pivot = named_poses[key] * 3
                    node_features[self.named_nodes[node_name], pivot] = x
                    node_features[self.named_nodes[node_name], pivot + 1] = y
                    node_features[self.named_nodes[node_name], pivot + 2] = z

                # edge feature
                for pivot, paper in enumerate(self.tape):
                    kor, eng, tags, func, pose = paper
                    edge_features[pivot * 2, named_poses[key]] = measure[i, pivot].item()
                    edge_features[pivot * 2, -1] = self.axis_code[func]
                    edge_features[pivot * 2, -2] = self.measure_code[func]

                    edge_features[pivot * 2 + 1, named_poses[key]] = measure[i, pivot].item()
                    edge_features[pivot * 2 + 1, -1] = self.axis_code[func]
                    edge_features[pivot * 2 + 1, -2] = self.measure_code[func]
            node_features = torch.tensor(node_features, dtype=torch.float32)
            edge_features = torch.tensor(edge_features, dtype=torch.float32)
            graph = Data(x=node_features,
                         edge_index=self.edge_indexes.contiguous(),
                         edge_attr=edge_features)
            if visualize:
                import networkx as nx
                import matplotlib.pyplot as plt

                # NetworkX 그래프로 변환
                G = nx.Graph()
                G.add_nodes_from(range(len(self.named_nodes)))
                G.add_edges_from(self.edge_indexes.t().tolist())

                # 노드 및 엣지 특성 추가
                for k in range(len(self.named_nodes)):
                    G.nodes[k]['features'] = node_features[k].numpy()

                for k, (src, tgt) in enumerate(self.edge_indexes.t().tolist()):
                    G[src][tgt]['features'] = edge_features[k].numpy()

                # 그래프 시각화
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=200, node_color='skyblue',
                        font_color='black', font_size=8)

                # 노드 및 엣지 특성 표시
                node_labels = nx.get_node_attributes(G, 'features')
                edge_labels = nx.get_edge_attributes(G, 'features')
                nx.draw_networkx_labels(G, pos, labels=node_labels)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                plt.show()
            graphs.append(graph)
        batch = Batch.from_data_list(graphs)
        return batch

if __name__ == "__main__":
    import numpy as np
    local_coordinates = np.array([
        [0.000004, 0.173072, 0.045842],
        [0.000004, 0.166068, 0.041727],
        [0.000005, 0.159199, 0.038891],
        [0.010534, 0.17442, 0.044278],
        [0.011509, 0.168968, 0.040661],
        [0.011514, 0.162216, 0.037741],
        [0.022467, 0.179147, 0.038456],
        [0.022797, 0.17234, 0.035668],
        [0.023112, 0.165165, 0.033298],
        [0.034549, 0.186121, 0.029206],
        [0.034367, 0.177863, 0.027833],
        [0.03448, 0.16983, 0.026763],
        [0.046961, 0.194257, 0.018282],
        [0.046277, 0.1841, 0.017975],
        [0.045782, 0.174998, 0.017928],
        [0.05668, 0.200305, 0.00533],
        [0.055841, 0.190139, 0.005745],
        [0.055365, 0.180477, 0.00599],
        [0.061498, 0.205366, -0.009416],
        [0.06096, 0.195296, -0.008638],
        [0.060575, 0.186406, -0.006652],
        [0.060109, 0.209468, -0.024662],
        [0.060253, 0.199692, -0.023644],
        [0.061064, 0.189445, -0.020463],
        [0.054943, 0.212463, -0.038293],
        [0.056568, 0.204191, -0.035638],
        [0.057695, 0.194955, -0.037318],
        [0.048084, 0.213733, -0.050248],
        [0.051125, 0.20588, -0.047239],
        [0.050769, 0.19784, -0.052408],
        [0.041848, 0.210761, -0.059673],
        [0.02917, 0.224814, -0.070147],
        [0.028967, 0.209895, -0.070632],
        [0.029091, 0.197508, -0.072452],
        [0.014155, 0.226973, -0.076392],
        [0.014635, 0.214205, -0.07585],
        [0.014823, 0.201555, -0.076463],
        [-0.000001, 0.227742, -0.077813],
        [0, 0.217802, -0.077588],
        [0.000001, 0.205623, -0.077723],
        [-0.014156, 0.226972, -0.076393],
        [-0.014635, 0.214203, -0.075851],
        [-0.014821, 0.201554, -0.076463],
        [-0.029171, 0.224811, -0.070148],
        [-0.028966, 0.209892, -0.070633],
        [-0.029089, 0.197505, -0.072454], # 어쩌면 누락필요
        [-0.043361, 0.221344, -0.056621],
        [-0.041847, 0.210756, -0.059674],
        [-0.04097, 0.199615, -0.063799],
        [-0.048084, 0.213728, -0.05025],
        [-0.052883, 0.222413, -0.042346],
        [-0.054945, 0.212458, -0.038295],
        [-0.056569, 0.204185, -0.035641],
        [-0.059988, 0.220752, -0.027262],
        [-0.060111, 0.209463, -0.024665],
        [-0.060254, 0.199686, -0.023647],
        [-0.062372, 0.217822, -0.011104],
        [-0.061494, 0.20536, -0.009418],
        [-0.060954, 0.19529, -0.00864],
        [-0.060568, 0.1864, -0.006654],
        [-0.056674, 0.2003, 0.005328],
        [-0.055834, 0.190132, 0.005743],
        [-0.055357, 0.180472, 0.005988],
        [-0.046954, 0.194252, 0.018281],
        [-0.046269, 0.184095, 0.017974],
        [-0.045773, 0.174993, 0.017926],
        [-0.034543, 0.186117, 0.029205],
        [-0.034358, 0.177859, 0.027832],
        [-0.03447, 0.169827, 0.026762],
        [-0.02246, 0.179144, 0.038456],
        [-0.022787, 0.172337, 0.035667],
        [-0.023102, 0.165163, 0.033297],
        [-0.010526, 0.174419, 0.044278],
        [-0.0115, 0.168967, 0.040661],
        [-0.011504, 0.162214, 0.03774],
    ])
    
    local_point = np.array([-0.131889, 0.08098, 0.040633])
    global_point = np.array([-0.131889, -0.040633, 0.08098])
    gap = global_point - local_point
    global_coordinates = local_coordinates + gap
    
    file_path = r"D:\Creadto\CreadtoLibrary\creadto-model\template\naked_body.obj"
    from creadto.utils.io import load_mesh
    vertex, face, _, _ = load_mesh(file_path)
    
    neck_indices = []
    for coordinate in local_coordinates:
        abs_error = abs(vertex - coordinate)
        sum_error = abs_error.sum(dim=1)
        print(sum_error.min())
        neck_indices.append(np.argmin(sum_error).item())
    neck_indices = np.array(neck_indices)
    
    import open3d as o3d
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(vertex.cpu().detach().numpy())
    color = np.zeros_like(vertex)
    color[neck_indices, 0] = 1.0
    mesh.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([mesh])
    
    np.save("./body_neck_indices.npy", neck_indices)