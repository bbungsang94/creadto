import inspect
import platform

import pandas as pd
import psutil
import torch
from torch_geometric.data import Data

from utils.log import print_message
from utils.visualize import draw_graph


def summary_graph(graph: Data, draw=False):
    print("=========SUMMARY GRAPH=========")
    print("CUDA: ", graph.is_cuda, " Undirected: ", graph.is_undirected(), " Coalesced ", graph.is_coalesced())
    print("=====Node:")
    print("# of node:", graph.num_nodes, "# of node feature", graph.num_node_features)
    print("node types: ", graph.num_node_types)
    print("=====Edges:")
    print("# of edges:", graph.num_edges, "# of edge feature", graph.num_edge_features)
    print("node types: ", graph.num_edge_types)
    print("=====Others:")

    if draw:
        draw_graph(graph)


def summary_device(bypass=False):
    # Write OS information
    print(print_message(message='', line='='))
    print(print_message(message='Operation System', padding=3, center=True, line=''))
    print(print_message(message=''))
    print(print_message(message='OS Name: ' + platform.system(), padding=2))
    print(print_message(message='OS Version: ' + platform.version(), padding=2))

    # Write processor information
    target = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(target)
    if target == 'cpu' or bypass:
        print(print_message(message='', line=''))
        print(print_message(message='CPU Setting(CUDA is not available)', padding=3, center=True, line=''))
        print(print_message(message=''))
        print(print_message(message='Process Architecture: ' + platform.machine(), padding=2))
        print(print_message(message='Process Details: ' + platform.processor(), padding=2))
        print(print_message(message='RAM Size: %.2f GB' % (psutil.virtual_memory().total / (1024 ** 3)), padding=2))
        print(print_message(message=''))
    else:
        properties = torch.cuda.get_device_properties(device)
        print(print_message(message='', line=''))
        print(print_message(message='GPU Setting(CUDA activated)', padding=3, center=True, line=''))
        print(print_message(message=''))
        print(print_message(message='Device Name: ' + properties.name, padding=2))
        print(print_message(message='Device Index: %02d' % torch.cuda.current_device(), padding=2))
        print(print_message(message='Device Memory: %.2f GB' % (properties.total_memory / (1024 ** 3)), padding=2))
        print(print_message(message='Device Major: %d' % properties.major, padding=2))
        print(print_message(message='Device Minor: %d' % properties.minor, padding=2))
        print(print_message(message='Device MP count: %d' % properties.multi_processor_count, padding=2))
        print(print_message(message='Device Is Integrated: ' + str(properties.is_integrated != 0), padding=2))
        print(print_message(message='Device Multi GPU Board: ' + str(properties.is_multi_gpu_board != 0), padding=2))
        print(print_message(message=''))
        log_torch_usage()

    print(print_message(message='', line=''))
    print(print_message(message='Creadto Inc.', padding=5, center=True))
    print(print_message(message='sangh.an@creadto.com', padding=5, center=True))
    print(print_message(message='', line='='))
    return device


def log_torch_usage(state_data: pd.DataFrame = None):
    state = torch.cuda.memory_stats()
    series = pd.Series(state)
    frame = series.to_frame().T
    if state_data is not None:
        state_data = pd.concat([state_data, frame], axis=0)
    else:
        state_data = frame
    return state_data


def get_return_variable_count(func):
    signature = inspect.signature(func)
    return len(signature.return_annotation)


def get_input_variable_count(func):
    signature = inspect.signature(func)
    return len(signature.parameters)