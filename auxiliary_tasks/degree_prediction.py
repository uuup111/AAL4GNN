import torch
from torch.nn import MSELoss
from torch_geometric.utils import degree


class DegreePredictionTask:
    def __init__(self, task_config):
        self.type = task_config['type']
        self.output_dim = task_config['params']['output_dim']
        self.loss_fn = MSELoss()

    def generate_node_degree(self, data):
        node_degree = degree(data.edge_index[0], num_nodes=data.num_nodes)
        return node_degree


