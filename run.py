import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.datasets import Planetoid

from auxiliary_tasks.auxiliary_config import add_auxiliary_task
from auxiliary_tasks.clustering import ClusteringTask
from model.multi_task_gcn import MultiTaskGCN
import yaml
from utils.data_loader import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='run_configs/config.yaml', help='Path to config file')
    return parser.parse_args()


def main():
    # load args
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Load dataset
    data, dataset = load_data(config)
    # add auxiliary tasks
    aux_tasks = add_auxiliary_task(config)
    # generate the auxiliary labels
    aux_labels = None
    for task in aux_tasks:
        if isinstance(task, ClusteringTask):
            aux_labels = task.generate_aux_labels(data)
    data.aux_labels = aux_labels
    # add model TODO: if we use meta-learning strategy, we may change it into the model array to get the model search space
    model = None
    if config['model']['name'] == 'gcn':
        model = MultiTaskGCN(data.num_features, dataset.num_classes, config['model']['params'], aux_tasks)

    # Params setting
    epochs = config['training']['epochs']
    acc = []
    alpha = 0.9
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()

    # 训练模型
    for epoch in range(0, epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        classification_loss = loss_fn(out[0][data.train_mask], data.y[data.train_mask])
        clustering_loss = loss_fn(out[1], data.aux_labels)
        loss = alpha * classification_loss + (1 - alpha) * clustering_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')

    # 评估模型
    model.eval()
    _, pred = model(data.x, data.edge_index)[0].max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    acc.append(accuracy)

    # 打印每个 alpha 对应的准确率
    for accuracy in acc:
        print(f'alpha: {alpha}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
