import copy

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data, Batch


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
            vis_idx = 520
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
