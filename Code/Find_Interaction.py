# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use GMM Algorithm to find out sample interaction/Cluster for Network based feature map method
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Calculate_Centrality
from sklearn.model_selection import train_test_split


# Import in data
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))
data_root = os.path.join(project_root, "Data", "Original Data")
data = pd.read_csv(data_root + '/dataset.csv')
features = data.iloc[:, 0:len(data.columns) - 1]
features_name = list(data.columns[:-1])
label = data.iloc[:, -1]

# ------------------------------------------划分数据集------------------------------------------------------
X_train, X_test, y_train, y_test = \
    train_test_split(features, label, test_size=0.33, shuffle=True, stratify=label, random_state=0)

# ------------------------------------------计算样本间interaction------------------------------------------------------
# Find Interaction using Network Science
# interaction_nodes_c = Calculate_Centrality.find_interaction(X_train, method="cosine", threshold=0.17, feature=False)
interaction_nodes_e = Calculate_Centrality.find_interaction(X_train, method="euclidean", threshold=10, feature=False)
# interaction_nodes_m = Calculate_Centrality.find_interaction(X_train, method="mahalanobis", threshold=0.3, feature=False)
"""
centrality_c, graph_info_c, nodes_pos_dict_c = Calculate_Centrality.draw_topology_graph(X_train, interaction_nodes_c,
                                                                                        layout='spring',
                                                                                        feature=False,
                                                                                        label=False)
"""

centrality_e, graph_info_e, nodes_pos_dict_e = Calculate_Centrality.draw_topology_graph(X_train, interaction_nodes_e, layout='spring', feature=False, label=False)

# centrality_m, graph_info_m, nodes_pos_dict_m = Calculate_Centrality.draw_topology_graph(X_train, interaction_nodes_m, layout='spring', feature=False, label=False)
