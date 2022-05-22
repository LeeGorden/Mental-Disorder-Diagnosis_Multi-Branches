# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use GCN Explainer to explain Graph Label
"""
import os
import pandas as pd
import numpy as np
from math import sqrt

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import k_hop_subgraph, to_networkx

import Calculate_Centrality
from Model_GNN_PyG import Net

EPS = 1e-15


class GNNGraphExplainer(torch.nn.Module):
    r"""
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    ★-PyG now just has class for Node Explainer, to apply GraphExplainer, Only A few functions we need to replace:
       1) we need to replace __subgraph__ function to obtain the computation graph for the entire graph.
       2) we need to set masks for the entire graph.
       3) We need to change the loss function to compute loss for graphs.
       即总结为以下:
       要学习的Mask作用在整个图上，不用取子图
       标签预测和损失函数的对象是单个graph
    """

    coeffs = {
        'edge_size': 0.001,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True, node=False):  # disable node_feat_mask by default
        super(GNNGraphExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.node = node

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        # 这里默认不学习node_feat_mask，要学习的话需在初始化GNNExplainer时将node置为True
        if self.node:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.edge_mask = torch.nn.Parameter(torch.zeros(E) * 50)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        if self.node:
            self.node_feat_masks = None
        self.edge_mask = None

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        if node_idx is not None:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow=self.__flow__())
            x = x[subset]
        else:
            x = x
            edge_index = edge_index
            row, col = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            edge_mask[:] = True
            mapping = None

        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def __graph_loss__(self, log_logits, pred_label):
        loss = -torch.log(log_logits[0, pred_label])
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx is not None:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            edge_mask = edge_mask[hard_edge_mask]
            subset = subset.tolist()
            if y is None:
                y = torch.zeros(edge_index.max().item() + 1,
                                device=edge_index.device)
            else:
                y = y[subset].to(torch.float) / y.max().item()
                y = y.tolist()
        else:
            subset = []
            for index, mask in enumerate(edge_mask):
                node_a = edge_index[0, index]
                node_b = edge_index[1, index]
                if node_a not in subset:
                    subset.append(node_a.item())
                if node_b not in subset:
                    subset.append(node_b.item())
            y = [y.cpu() for i in range(len(subset))]  # Need to convert tensor y from cuda to cpu first

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=len(y)).to('cpu')
        G = to_networkx(data, edge_attrs=['att'])  # , node_attrs=['y']
        mapping = {k: i for k, i in enumerate(subset)}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 800
        kwargs['cmap'] = kwargs.get('cmap') or 'Purples'

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.7),
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        # nx.draw_networkx_nodes(G, pos, node_color=y, **kwargs)
        nx.draw_networkx_nodes(G, pos, node_color="y", label=True, node_size=800, cmap="Purples")
        # nx.draw_networkx_labels(G, pos, **kwargs)
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.show(block=True)
        return ax, G

    def explain_graph(self, data, **kwargs):
        self.model.eval()
        self.__clear_masks__()
        x, edge_index, batch = data.x, data.edge_index, data.batch

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, _, hard_edge_mask, kwargs = self.__subgraph__(node_idx=None, x=x, edge_index=edge_index,
                                                                     **kwargs)
        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(data, **kwargs)
            probs_Y = torch.softmax(log_logits, 1)
            pred_label = probs_Y.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        if self.node:
            optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                         lr=self.lr)
        else:
            optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        epoch_losses = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            optimizer.zero_grad()
            if self.node:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()

            log_logits = self.model(data, **kwargs)
            pred = torch.softmax(log_logits, 1)
            loss = self.__graph_loss__(pred, pred_label)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_losses.append(epoch_loss)

        edge_mask = self.edge_mask.detach().sigmoid()
        print(edge_mask)

        self.__clear_masks__()

        return edge_mask, epoch_losses

    def __repr__(self):
        return f'{self.__class__.__name__}()'


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
# # Create data for test set
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

node_features = torch.tensor(np.array(X_test), dtype=torch.float)
target = torch.tensor(np.array(y_test.iloc[:, 1]), dtype=torch.float).unsqueeze(dim=1)
GraphData_list = list()
for i in range(len(node_features)):
    GraphData_list.append(Data(x=node_features[i].unsqueeze(dim=1), y=target[i].unsqueeze(dim=0),
                               edge_index=edge_index, edge_attr=weight_of_edges))
test_loader = DataLoader(GraphData_list, batch_size=1)
# 统计以便计算每个epoch的平均准确率
test_num = len(GraphData_list)

model = Net(1, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load('GNN_w_PyG_D1.pth')
model.load_state_dict(state)
model.to(device)

for data in test_loader:
    explainer = GNNGraphExplainer(model, epochs=200)
    data = data.to(device)
    edge_mask, _ = explainer.explain_graph(data)
    ax, G = explainer.visualize_subgraph(None, data.edge_index, edge_mask, y=data.y)
    plt.show(block=True)
    break