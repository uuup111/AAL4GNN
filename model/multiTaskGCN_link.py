import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges


class MultiTaskGCN_link(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_tasks=2):
        super(MultiTaskGCN_link, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        # self.shared_representation = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

        # 边预测的线性层
        self.edge_predictor = torch.nn.Linear(num_classes, 1)

        self.num_tasks = num_tasks

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index=None):
        """
        前向传播，整合节点分类和边预测。
        x: 节点特征
        edge_index: 图的边索引
        pos_edge_index: 正样本边索引
        neg_edge_index: 负样本边索引（用于边预测）
        """
        # 1. 节点分类的输出
        node_embeddings = self.node_classification(x, edge_index)

        # 2. 边预测任务 (基于节点的嵌入)
        pos_edge_logits = self.edge_prediction(
            self.get_edge_features(node_embeddings, pos_edge_index)
        )

        # 处理负样本边（如果提供了负样本边索引）
        if neg_edge_index is not None:
            neg_edge_logits = self.edge_prediction(
                self.get_edge_features(node_embeddings, neg_edge_index)
            )
            edge_prediction_output = torch.cat([pos_edge_logits, neg_edge_logits], dim=0)
        else:
            edge_prediction_output = pos_edge_logits

        return node_embeddings, edge_prediction_output

    def node_classification(self, x, edge_index):
        """
        节点分类的前向传播部分
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def edge_prediction(self, z):
        """
        边预测部分，z 是从节点嵌入中提取的边的特征
        """
        z = self.edge_predictor(z)
        return z

    def get_edge_features(self, node_embeddings, edge_index):
        """
        从节点嵌入中提取边的特征
        这里我们将每条边的两个节点的特征拼接在一起作为边的特征
        """
        source = node_embeddings[edge_index[0]]
        target = node_embeddings[edge_index[1]]
        edge_features = torch.cat([source, target], dim=1)  # 拼接源节点和目标节点的嵌入
        return edge_features

    def compute_losses(self, outputs, labels, pos_edge_index, neg_edge_index):
        """
        计算节点分类损失和边预测损失
        """
        node_classification_loss = torch.nn.functional.cross_entropy(outputs[0], labels)

        # 边预测损失 (正样本为1，负样本为0)
        edge_prediction_logits = outputs[1].squeeze()  # 去除单维度
        pos_labels = pos_edge_index.new_ones(pos_edge_index.size(1))  # 正样本标签为1
        neg_labels = neg_edge_index.new_zeros(neg_edge_index.size(1))  # 负样本标签为0
        edge_labels = torch.cat([pos_labels, neg_labels], dim=0)  # 合并正负样本标签

        edge_prediction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            edge_prediction_logits, edge_labels
        )
        return node_classification_loss, edge_prediction_loss