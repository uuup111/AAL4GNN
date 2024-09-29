import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric import transforms
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling
from model.multiTaskGCN_link import MultiTaskGCN_link

# 初始化数据、模型、损失函数和优化器
dataset = Planetoid(root='./data/', name='Cora')
data = dataset[0]
data = transforms.RandomLinkSplit(data)  # 创建训练集和测试集的边索引

# 评估多任务学习的不同权重 alpha
alpha = 0.8

model = MultiTaskGCN_link(data.num_features, dataset.num_classes)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

# 训练模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()

    # 前向传播
    node_embeddings, edge_prediction_output = model(
        data.x, data.train_pos_edge_index, data.train_pos_edge_index, data.train_neg_edge_index
    )

    # 计算节点分类和边预测的损失
    node_classification_loss, edge_prediction_loss = model.compute_losses(
        (node_embeddings, edge_prediction_output), data.y[data.train_mask],
        data.train_pos_edge_index, data.train_neg_edge_index
    )

    # 加权损失总和
    total_loss = alpha * node_classification_loss + (1 - alpha) * edge_prediction_loss
    total_loss.backward()
    optimizer.step()

    print(f'Epoch {epoch:03d}, Loss: {total_loss.item():.4f}')

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    node_embeddings, _ = model(data.x, data.test_pos_edge_index, data.test_pos_edge_index)
    _, pred = node_embeddings.max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    print(f'Accuracy: {accuracy:.4f}')
