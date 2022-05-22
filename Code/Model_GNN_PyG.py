# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

GCN
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool


class Net(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        # GCN layers
        self.conv1 = GATConv(in_dim, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = GATConv(32, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.conv3 = GATConv(64, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.conv4 = GATConv(128, 64)
        self.bn4 = torch.nn.BatchNorm1d(64)

        # Output layer
        self.fc1 = Linear(64, 32)
        self.classify = Linear(32, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # ★这里的data.x包含大图(包含若干个小图)的node feature
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index))
        x = self.bn4(x)

        x = global_add_pool(x, batch)  # 这一步必要, 表示以add形式aggregate
        x = F.leaky_relu(self.fc1(x))
        return torch.sigmoid(self.classify(x))
