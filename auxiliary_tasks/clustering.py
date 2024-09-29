# tasks/clustering.py
import torch
from sklearn.cluster import KMeans


class ClusteringTask:
    def __init__(self, task_config):
        self.type = task_config['type']
        self.method = task_config['params']['method']
        self.n_clusters = task_config['params']['n_clusters']
        if self.method == 'Kmeans':
            self.cluster = KMeans(n_clusters=self.n_clusters)

    def generate_aux_labels(self, data):
        aux_labels = self.cluster.fit_predict(data.x.numpy())
        return torch.tensor(aux_labels, dtype=torch.long)
