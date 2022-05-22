# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use GCN to conduct prediction for the property of the whole graph
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

import Calculate_Centrality
from Model_GNN_PyG import Net

from torch_geometric.data import InMemoryDataset, download_url


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, features_node, target_to_predict, edges, edges_attr=None,
                 transform=None, pre_transform=None):
        self.features_node = features_node
        self.target_to_predict = target_to_predict
        self.edges = edges
        self.edges_attr = edges_attr

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    # 返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']

    # 用于从网上下载数据集
    """
    由于是本地数据集所以不需要download;
    如果不注释在初始的时候回自动先执行download方法，然后执行processed_file_names方法返回本地的.pt文件并重构torch_geometric.data
    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...
    """

    # 生成数据集所用的方法
    """
    为了填充data_list里的内容，需要先生成data对象。
    在PyG中，单个graph定义为torch_geometric.data.Data实例，默认有以下属性：
    - data.x：节点特征矩阵，shape为[num_nodes, num_node_features]。
    - data.edge_index：COO格式的graph connectivity矩阵，shape为[2, num_edges]，类型为torch.long。
    - data.edge_attr：边的特征矩阵，shape为[num_edges, num_edge_features]。
    - data.y：训练的target，shape不固定，比如，对于node-level任务，形状为[num_nodes, *]，对于graph-level任务，形状为[1, *],
      而对于节点任务来说形状可以是[[node_num1,target],[node_num2,target],…],其中target可以为数字或者one-hot等。
    - data.pos：节点的位置(position)矩阵
    """
    def process(self):
        PyG_GraphData = Data(x=self.features_node, y=self.target_to_predict,
                        edge_index=self.edges, edge_attr=self.edges_attr)
        # Read data into huge `Data` list.
        # ★ If there are more than 1 graph, should do data_list = [PyG_GraphData1, PyG_GraphData2, PyG_GraphData3...]
        data_list = [PyG_GraphData]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# -------------------------------------------Preparing Data----------------------------------------------
# Import in data
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_root = os.path.join(project_root, "Data", "Original Data")
data = pd.read_csv(data_root + '/multilabel.csv')
features = data.iloc[:, :-11]
features_name = list(features.columns)
labels = data.iloc[:, -11:]
labels_name = list(labels.columns)

# ------------------------------------------划分数据集------------------------------------------------------
X_train, X_test, y_train, y_test = \
    train_test_split(features, labels, test_size=0.33, shuffle=True, stratify=labels.iloc[:, 0], random_state=0)

# ------------------------------------------Create Own Dataset--------------------------------------------------
# Create data for train set
# preparing data for edge_data, since it's binary/non direction graph, need to duplicate src, tgt, weights
"""
data.x：节点特征矩阵，shape为[num_nodes, num_node_features]
"""
interaction_nodes, edge_weights = Calculate_Centrality.find_interaction(X_train, method='cosine', threshold=0.12)
src = [interaction_nodes[i][0] for i in range(len(interaction_nodes))]
tgt = [interaction_nodes[i][1] for i in range(len(interaction_nodes))]
# If you were to create dataset for non-direction graph, just need to add repeated edge like below
"""
data.edge_index: COO格式的graph connectivity矩阵，shape为[2, num_edges]，类型为torch.long。
"""
edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)

"""
data.edge_attr: 边的特征矩阵，shape为[num_edges, num_edge_features], 顺序为edge_index的顺序
"""
weight_of_edges = torch.tensor([edge_weights + edge_weights], dtype=torch.float).permute(1, 0)

# preparing data for node_data, it is the feature of each node
"""
data.x: 节点特征矩阵，shape为[num_nodes, num_node_features]
"""
node_features = torch.tensor(np.array(X_train), dtype=torch.float)
# a = node_features[0].unsqueeze(dim=1)

# preparing labels to be predicted
"""
data.y: 训练的target，shape不固定，比如，对于node-level任务，形状为[num_nodes, *]，对于graph-level任务，形状为[1, *],
而对于节点任务来说形状可以是[[node_num1,target],[node_num2,target],…],其中target可以为数字或者one-hot等。
"""
target = torch.tensor(np.array(y_train), dtype=torch.float)
# a = target[0].unsqueeze(dim=0)

# Compile into graph dataset and check info
"""
PyG_GraphData_0 = Data(x=node_features[0].unsqueeze(dim=1), y=target[0].unsqueeze(dim=0),
                       edge_index=edge_index, edge_attr=weight_of_edges)
PyG_GraphData_0.num_nodes
PyG_GraphData_0.num_edges
PyG_GraphData_0.contains_isolated_nodes()
PyG_GraphData_0.contains_self_loops()
PyG_GraphData_0.is_directed()
PyG_GraphData_0.edge_attr
PyG_GraphData_0.y
"""
GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
train_loader = DataLoader(GraphData_list, batch_size=32)
# 统计以便计算每个epoch的平均准确率
train_num = len(GraphData_list)

# Create data for test set
node_features = torch.tensor(np.array(X_test), dtype=torch.float)
target = torch.tensor(np.array(y_test), dtype=torch.float)
GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
test_loader = DataLoader(GraphData_list, batch_size=32)
# 统计以便计算每个epoch的平均准确率
test_num = len(GraphData_list)
"""
"MyOwnDataset" and "Dataset" of PyG are used to create local file and perform transform;
if no need to create local file or perform transform, can direct use dataloader to generate graph data.
b = MyOwnDataset(root="../Data/PyG_GraphData", features_node=node_features, target_to_predict=target,
                 edges=edge_index, edges_attr=weight_of_edges)
"""
# train the model
# ------------------------------------同时训练11个label的多分类模型---------------------------------------
# 设置模型参数
# 设置in_dim, hidden_dim, out_dim
model = Net(1, 11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load('GNN_w_PyG.pth')
model.load_state_dict(state)

model.to(device)
# 定义分类交叉熵损失
loss_func = nn.BCELoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义循环次数
epochs = 500
# 定义保存路径
save_path = "GNN_w_PyG.pth"


# 定义多标签分来下准确率的计算
def calculate_prediction(model_predicted, accuracy_th=0.5):
    # 注意这里的model_predicted是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    predicted_result = model_predicted > accuracy_th  # tensor之间比较大小跟array一样
    predicted_result = predicted_result.float()  # 因为label是0, 1的float, 所以这里需要将他们转化为同类型tensor

    return predicted_result


# 模型训练
train_losses = list()
test_losses = list()
best_acc = 0.0
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    # train
    model.train()
    train_loss = 0.0
    # 记录每个标签加总准确率
    train_acc_all_labels = 0.0
    # 记录每个标签的各自准确率, 这里有len(train_loader.dataset.labels[0])个标签
    train_acc_separate_labels = torch.tensor([0 for _ in range(len(labels_name))]).float()
    train_acc_separate_labels = train_acc_separate_labels.to(device)
    train_bar = tqdm(train_loader)
    for batch_idx, batched_graph in enumerate(train_bar):
        batched_graph = batched_graph.to(device)
        # ★这里一定将labels转化为labels.float()转化为32位否则无法与prediction(float32)匹配进行计算loss
        labels = batched_graph.y.float().to(device)
        predicted_prob = model(batched_graph)
        # 将概率输出x_hat转化为标签变量0, 1
        predicted_y = calculate_prediction(predicted_prob, accuracy_th=0.5)  # 由于是多标签分类, 这里的predict_y是一个11维tensor
        # torch.eq(input, other, *, out=None)比较两张量是否相同。 计算一个batch 每个label平均准确率之和
        train_acc_all_labels += torch.eq(predicted_y, labels).sum().item() / labels.size()[1]
        # .sum(dim=0)是将输出的label命中率按tensor batch size求和(这里是第一维所以sum(dim=0)
        train_acc_separate_labels += torch.eq(predicted_y, labels).float().sum(dim=0)  # 如果不把bool转为float, 会变成逻辑运算
        loss = loss_func(predicted_prob, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录到目前为止所有batch的loss总和
        train_loss += loss.item()
        # 显示当前batch loss
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # train完一个epoch统计accuracy和loss
    train_accurate_all_labels = train_acc_all_labels / train_num
    train_accurate_separate_labels = train_acc_separate_labels / train_num
    train_loss /= (batch_idx + 1)
    train_losses.append(train_loss)

    # test
    model.eval()
    test_loss = 0.0
    test_acc_all_labels = 0.0
    test_acc_separate_labels = torch.tensor([0 for _ in range(len(labels_name))]).float()
    test_acc_separate_labels = test_acc_separate_labels.to(device)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, batched_graph in enumerate(test_bar):
            batched_graph = batched_graph.to(device)
            labels = batched_graph.y.float().to(device)
            predicted_prob = model(batched_graph)
            predicted_y = calculate_prediction(predicted_prob, accuracy_th=0.5)
            test_acc_all_labels += torch.eq(predicted_y, labels).sum().item() / labels.size()[1]
            test_acc_separate_labels += torch.eq(predicted_y, labels).float().sum(dim=0)
            loss = loss_func(predicted_prob, labels)
            test_loss += loss
            test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)
    # test完一个epoch统计accuracy
    test_accurate_all_labels = test_acc_all_labels / test_num
    test_accurate_separate_labels = test_acc_separate_labels / test_num
    test_loss /= (batch_idx + 1)
    test_losses.append(test_loss)

    # 输出一个epoch的所有数据
    print('[epoch %d]' % (epoch + 1))
    print('train_loss: %.3f; test_loss: %.3f' % (train_loss, test_loss))
    print('train_accuracy_all_labels:')
    print(train_accurate_all_labels)
    print('train_accuracy_separate_labels:')
    print(train_accurate_separate_labels)
    print('val_accuracy_all_labels:')
    print(test_accurate_all_labels)
    print('val_accuracy_separate_labels:')
    print(test_accurate_separate_labels)

    if test_accurate_all_labels > best_acc:
        best_acc = test_accurate_all_labels
        best_acc_separate_labels = test_accurate_separate_labels
        torch.save(model.state_dict(), save_path)
        print('Model Saved')

# ------------------------------------同时训练Positive&Negative模型---------------------------------------
interaction_nodes, edge_weights = Calculate_Centrality.find_interaction(X_train, method='cosine', threshold=0.12)
src = [interaction_nodes[i][0] for i in range(len(interaction_nodes))]
tgt = [interaction_nodes[i][1] for i in range(len(interaction_nodes))]

edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
weight_of_edges = torch.tensor([edge_weights + edge_weights], dtype=torch.float).permute(1, 0)

node_features = torch.tensor(np.array(X_train), dtype=torch.float)
target = torch.tensor(np.array(y_train.iloc[:, 0]), dtype=torch.float).unsqueeze(dim=1)

GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
train_loader = DataLoader(GraphData_list, batch_size=32)
# 统计以便计算每个epoch的平均准确率
train_num = len(GraphData_list)

# Create data for test set
node_features = torch.tensor(np.array(X_test), dtype=torch.float)
target = torch.tensor(np.array(y_test.iloc[:, 0]), dtype=torch.float).unsqueeze(dim=1)
GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
test_loader = DataLoader(GraphData_list, batch_size=32)
# 统计以便计算每个epoch的平均准确率
test_num = len(GraphData_list)

# 设置模型参数
# 设置in_dim, hidden_dim, out_dim
model = Net(1, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load('GNN_w_PyG_P&N.pth')
model.load_state_dict(state)

model.to(device)
# 定义分类交叉熵损失
loss_func = nn.BCELoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义循环次数
epochs = 500
# 定义保存路径
save_path = "GNN_w_PyG_P&N.pth"


# 定义多标签分来下准确率的计算
def calculate_prediction(model_predicted, accuracy_th=0.5):
    # 注意这里的model_predicted是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    predicted_result = model_predicted > accuracy_th  # tensor之间比较大小跟array一样
    predicted_result = predicted_result.float()  # 因为label是0, 1的float, 所以这里需要将他们转化为同类型tensor

    return predicted_result


# 模型训练
train_losses = list()
test_losses = list()
best_acc = 0.0
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    # train
    model.train()
    train_loss = 0.0
    # 记录每个标签加总准确率
    train_acc_all_labels = 0.0
    train_bar = tqdm(train_loader)
    for batch_idx, batched_graph in enumerate(train_bar):
        batched_graph = batched_graph.to(device)
        # ★这里一定将labels转化为labels.float()转化为32位否则无法与prediction(float32)匹配进行计算loss
        labels = batched_graph.y.float().to(device)
        predicted_prob = model(batched_graph)
        # 将概率输出x_hat转化为标签变量0, 1
        predicted_y = calculate_prediction(predicted_prob, accuracy_th=0.5)  # 由于是多标签分类, 这里的predict_y是一个11维tensor
        # torch.eq(input, other, *, out=None)比较两张量是否相同。 计算一个batch 每个label平均准确率之和
        train_acc_all_labels += torch.eq(predicted_y, labels).sum().item()
        loss = loss_func(predicted_prob, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录到目前为止所有batch的loss总和
        train_loss += loss.item()
        # 显示当前batch loss
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # train完一个epoch统计accuracy和loss
    train_accurate_all_labels = train_acc_all_labels / train_num
    train_loss /= (batch_idx + 1)
    train_losses.append(train_loss)

    # test
    model.eval()
    test_loss = 0.0
    test_acc_all_labels = 0.0
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, batched_graph in enumerate(test_bar):
            batched_graph = batched_graph.to(device)
            labels = batched_graph.y.float().to(device)
            predicted_prob = model(batched_graph)
            predicted_y = calculate_prediction(predicted_prob, accuracy_th=0.5)
            test_acc_all_labels += torch.eq(predicted_y, labels).sum().item()
            loss = loss_func(predicted_prob, labels)
            test_loss += loss
            test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)
    # test完一个epoch统计accuracy
    test_accurate_all_labels = test_acc_all_labels / test_num
    test_loss /= (batch_idx + 1)
    test_losses.append(test_loss)

    # 输出一个epoch的所有数据
    print('[epoch %d]' % (epoch + 1))
    print('train_loss: %.3f; test_loss: %.3f' % (train_loss, test_loss))
    print('train_accuracy_all_labels:')
    print(train_accurate_all_labels)
    print('val_accuracy_all_labels:')
    print(test_accurate_all_labels)

    if test_accurate_all_labels > best_acc:
        best_acc = test_accurate_all_labels
        torch.save(model.state_dict(), save_path)
        print('Model Saved')

# ------------------------------------D1---------------------------------------
interaction_nodes, edge_weights = Calculate_Centrality.find_interaction(X_train, method='cosine', threshold=0.12)
src = [interaction_nodes[i][0] for i in range(len(interaction_nodes))]
tgt = [interaction_nodes[i][1] for i in range(len(interaction_nodes))]

edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
weight_of_edges = torch.tensor([edge_weights + edge_weights], dtype=torch.float).permute(1, 0)

node_features = torch.tensor(np.array(X_train), dtype=torch.float)
target = torch.tensor(np.array(y_train.iloc[:, 1]), dtype=torch.float).unsqueeze(dim=1)

GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
train_loader = DataLoader(GraphData_list, batch_size=32)
# 统计以便计算每个epoch的平均准确率
train_num = len(GraphData_list)

# Create data for test set
node_features = torch.tensor(np.array(X_test), dtype=torch.float)
target = torch.tensor(np.array(y_test.iloc[:, 1]), dtype=torch.float).unsqueeze(dim=1)
GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
test_loader = DataLoader(GraphData_list, batch_size=32)
# 统计以便计算每个epoch的平均准确率
test_num = len(GraphData_list)

# 设置模型参数
# 设置in_dim, hidden_dim, out_dim
model = Net(1, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# 定义分类交叉熵损失
loss_func = nn.BCELoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义循环次数
epochs = 500
# 定义保存路径
save_path = "GNN_w_PyG_D1.pth"


# 定义多标签分来下准确率的计算
def calculate_prediction(model_predicted, accuracy_th=0.5):
    # 注意这里的model_predicted是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    predicted_result = model_predicted > accuracy_th  # tensor之间比较大小跟array一样
    predicted_result = predicted_result.float()  # 因为label是0, 1的float, 所以这里需要将他们转化为同类型tensor

    return predicted_result


# 模型训练
train_losses = list()
test_losses = list()
best_acc = 0.0
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    # train
    model.train()
    train_loss = 0.0
    # 记录每个标签加总准确率
    train_acc_all_labels = 0.0
    train_bar = tqdm(train_loader)
    for batch_idx, batched_graph in enumerate(train_bar):
        batched_graph = batched_graph.to(device)
        # ★这里一定将labels转化为labels.float()转化为32位否则无法与prediction(float32)匹配进行计算loss
        labels = batched_graph.y.float().to(device)
        predicted_prob = model(batched_graph)
        # 将概率输出x_hat转化为标签变量0, 1
        predicted_y = calculate_prediction(predicted_prob, accuracy_th=0.5)  # 由于是多标签分类, 这里的predict_y是一个11维tensor
        # torch.eq(input, other, *, out=None)比较两张量是否相同。 计算一个batch 每个label平均准确率之和
        train_acc_all_labels += torch.eq(predicted_y, labels).sum().item()
        loss = loss_func(predicted_prob, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录到目前为止所有batch的loss总和
        train_loss += loss.item()
        # 显示当前batch loss
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # train完一个epoch统计accuracy和loss
    train_accurate_all_labels = train_acc_all_labels / train_num
    train_loss /= (batch_idx + 1)
    train_losses.append(train_loss)

    # test
    model.eval()
    test_loss = 0.0
    test_acc_all_labels = 0.0
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, batched_graph in enumerate(test_bar):
            batched_graph = batched_graph.to(device)
            labels = batched_graph.y.float().to(device)
            predicted_prob = model(batched_graph)
            predicted_y = calculate_prediction(predicted_prob, accuracy_th=0.5)
            test_acc_all_labels += torch.eq(predicted_y, labels).sum().item()
            loss = loss_func(predicted_prob, labels)
            test_loss += loss
            test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)
    # test完一个epoch统计accuracy
    test_accurate_all_labels = test_acc_all_labels / test_num
    test_loss /= (batch_idx + 1)
    test_losses.append(test_loss)

    # 输出一个epoch的所有数据
    print('[epoch %d]' % (epoch + 1))
    print('train_loss: %.3f; test_loss: %.3f' % (train_loss, test_loss))
    print('train_accuracy_all_labels:')
    print(train_accurate_all_labels)
    print('val_accuracy_all_labels:')
    print(test_accurate_all_labels)

    if test_accurate_all_labels > best_acc:
        best_acc = test_accurate_all_labels
        torch.save(model.state_dict(), save_path)
        print('Model Saved')
