# tasks/clustering.py
import torch
from sklearn.cluster import KMeans
from torch.nn import CrossEntropyLoss


class ClusteringTask:
    def __init__(self, task_config):
        self.type = task_config['type']
        self.method = task_config['params']['method']
        self.output_dim = task_config['params']['output_dim']
        self.loss_fn = CrossEntropyLoss()
        if self.method == 'Kmeans':
            self.cluster = KMeans(n_clusters=self.output_dim)

    def generate_aux_labels(self, data):
        aux_labels = self.cluster.fit_predict(data.x.numpy())
        return torch.tensor(aux_labels, dtype=torch.long)
