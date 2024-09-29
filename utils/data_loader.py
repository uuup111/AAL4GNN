# utils/data_loader.py
import numpy as np
import torch
from torch_geometric.datasets import Planetoid


def load_data(config):
    data = None
    dataset = None
    if config['dataset']['name'] == 'Cora':
        data, dataset = load_Cora_data(config)
    return data, dataset


def parse_cora(path):
    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                             'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning',
                                             'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, np.unique(data_edges, axis=0).transpose()


def load_Cora_data(config):
    dataset = Planetoid(root='./data/', name=config['dataset']['name'])
    data = dataset[0]
    data_X, data_Y, data_edges = parse_cora(config['dataset']['params']['path'])
    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # Split data
    train_ratio = config['split']['train_ratio']
    val_ratio = config['split']['val_ratio']
    test_ratio = config['split']['test_ratio']

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    train_end = int(data.num_nodes * train_ratio)
    val_end = train_end + int(data.num_nodes * val_ratio)

    data.train_id = np.sort(node_id[:train_end])
    data.val_id = np.sort(node_id[train_end:val_end])
    data.test_id = np.sort(node_id[val_end:])

    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

    return data, dataset
