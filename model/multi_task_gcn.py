import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from torch.nn import CrossEntropyLoss, ModuleList
from torch.optim import Adam
import torch.nn.functional as F


# 定义多任务GCN模型
class MultiTaskGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, model_params, aux_tasks):
        super(MultiTaskGCN, self).__init__()
        self.hidden_channels = model_params['hidden_channels']
        self.drop_rate = model_params['drop_rate']
        # 定义图卷积层
        self.convs = ModuleList([
            GCNConv(num_features, self.hidden_channels),  # 第一层图卷积
            # GCNConv(hidden_channels, hidden_channels),  # 第二层图卷积
        ])
        # define the classifier head
        self.classifier = GCNConv(self.hidden_channels, num_classes)
        # define the auxiliary head by aux_tasks
        self.cluster_head = None
        self.auxiliary_tasks = {}
        for aux_task in aux_tasks:
            if aux_task.type == 'clustering':
                self.auxiliary_tasks[aux_task.type] = GCNConv(self.hidden_channels, aux_task.output_dim)
            elif aux_task.type == 'degree_prediction':
                # TODO: Linear or GCN ? That's a question
                self.auxiliary_tasks[aux_task.type] = GCNConv(self.hidden_channels, aux_task.output_dim)

    def forward(self, x, edge_index):
        # 前向传播过程
        x = F.relu(self.convs[0](x, edge_index))  # 第一层图卷积
        x = F.dropout(x, p=self.drop_rate, training=self.training)  # dropout正则化
        # x = F.relu(self.convs[1](x, edge_index))  # 第二层图卷积
        # 分类头的输出
        main_out = self.classifier(x, edge_index)
        # the output of auxiliary tasks
        auxiliary_out = {}
        for task in self.auxiliary_tasks:
            auxiliary_out[task] = self.auxiliary_tasks[task](x, edge_index)
        return main_out, auxiliary_out
