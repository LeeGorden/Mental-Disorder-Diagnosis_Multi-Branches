# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Calculating different centrality in Network Science
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import Hamming_Distance


# ------------------------Calculation of Similarity and Definition of Interaction--------------------------
def find_interaction(x, method='cosine', threshold=0.2, feature=True):
    """
    Function:
    find the interaction between features
    Variable:
    x-DataFrame of raw data with row = samples and col = features(excluding y-label)
    method-{'circle', 'random', 'shell', 'spring', 'spectral'}
           the method to calculate distance between samples/features in input x
    threshold-the threshold of distance to be defined as 'has interaction'
    feature-if = True, meaning to calculating distance among features, else among samples
    Return:
    list of sample index/feature names of x which have interaction
    """
    # Initialize variable to be used
    interact_list = list()
    # When feature=True, find interaction between features, else find interaction between samples
    if feature:
        nodes_name = x.columns
        x = x.T
    else:
        nodes_name = x.index
    # Calculate the distance based on method input
    if method == "cosine":
        print('Calculating distance using cosine distance')
        dist = pdist(x, metric='cosine')
    elif method == "euclidean":
        print('Calculating distance using euclidean distance')
        dist = pdist(x, metric='euclidean')
    elif method == "mahalanobis":
        print('Calculating distance using mahalanobis distance')
        dist = pdist(x, metric='mahalanobis')
    elif method == 'hamming':
        print('Calculating distance using hamming distance')
        dist = Hamming_Distance.hamming_distance_32bits(np.array(x))
    dist_max = np.max(dist)
    dist_min = np.min(dist)
    # Draw distribution chart of cosine distance
    plt.figure()
    plt.hist(dist)
    plt.title(method + ' distance')
    plt.xlabel(method + ' distance')
    plt.ylabel('count')
    plt.annotate('Mean: ' + str(round(np.mean(dist), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 1),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 1))
    plt.annotate('Max: ' + str(round(dist_max, 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.9),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.9))
    plt.annotate('75%: ' + str(round(np.percentile(dist, 75), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.8),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.8))
    plt.annotate('50%: ' + str(round(np.percentile(dist, 50), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.7),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.7))
    plt.annotate('25%: ' + str(round(np.percentile(dist, 25), 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.6),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.6))
    plt.annotate('Min: ' + str(round(dist_min, 3)), xy=(0.75 * dist_max, len(dist) / 5 * 0.5),
                 xytext=(0.75 * dist_max, len(dist) / 5 * 0.5))
    plt.show()
    # Select object satisfying the threshold, define them as 'has interaction' and record in a list
    dist = squareform(dist)
    for i in range(0, len(dist)):
        for j in range(i + 1, len(dist)):
            # 当距离满足条件的时候, 将对应feature_name放到interact_list中
            if dist[i][j] <= threshold:
                # Since we are going to create edges for a DiGraph using interact_list
                # Thus put both [i, j] and [j, i] into interact_list
                interact_list.append([nodes_name[i], nodes_name[j]])
    return interact_list


# -----------------------------Draw Topology Graph-------------------------------------------
def draw_topology_graph(x, edges, layout='random', feature=True, label=True):
    """
    Function:
    Return the topology graph and corresponding centrality
    Variable:
    x-DataFrame, data input
    edges-edges of the graph defined as 'has interaction'
    layout-g.pos, type of topology used
    feature-if = True, meaning to calculating distance among features, else among samples
    Return:
    graph and centrality features
    """
    # Initialize variable to be used
    if feature:
        nodes_name = x.columns
        x = x.T
    else:
        nodes_name = x.index
    # 创建无向图, 因为不是sequence data,
    # 且这里不存在一个feature指向另一个feature而另一个feature与该feature无interaction的情况
    g = nx.Graph()  # 创建有向图g = nx.DiGraph()
    # Way to add node one-by-one: g.add_node(0)
    # Add_nodes_from([])可以将list内的节点加入
    g.add_nodes_from(nodes_name)
    # 为node增加data, 将每个feature对应所有samples的值放入node中, 通过g.nodes[i]读取存储进去的数据
    for node_name in g.nodes():
        g.nodes[node_name].update(x.loc[node_name])
    # Graph边的信息即两两有interaction的样本点(这里由于是有向图且不是时间序列所以是排列)
    g.add_edges_from(edges)
    # Set layout for network graph
    if layout == 'random':
        g.pos = nx.random_layout(g)  # default to scale=1
    elif layout == 'spring':
        g.pos = nx.spring_layout(g, iterations=200)
    elif layout == 'shell':
        g.pos = nx.shell_layout(g)
    elif layout == 'circle':
        g.pos = nx.circular_layout(g)
    elif layout == 'spectral':
        g.pos = nx.spectral_layout(g)
    elif layout == 'kamada_kawai':
        g.pos = nx.kamada_kawai_layout(g)
    # Plot the network graph
    plt.figure()
    nx.draw(g, pos=g.pos,
            node_size=100, node_color='y', with_labels=label, font_size=12,
            alpha=0.5, width=0, style='solid')
    plt.title('Feature Network_' + layout)
    plt.show()
    # Output centrality information
    # Degree_centrality描述节点与其他节点联系强弱, =连到该节点的edge/总edge数
    degree_c = pd.DataFrame({'Degree Centrality': nx.degree_centrality(g)})
    # Closeness_centrality描述节点靠近网络中心程度, =(n-1) * 1/该节点沿着edge走到其他所有节点距离之和(两个被连线节点的距离是1)
    closeness_c = pd.DataFrame({'Closeness Centrality': nx.closeness_centrality(g)})
    # Betweenness_centrality表述节点被经过的次数 ,=至少有三个点的路线出现的次数
    betweenness_c = pd.DataFrame({'Betweeness Centrality': nx.betweenness_centrality(g)})
    load_c = pd.DataFrame({'Load Centrality': nx.load_centrality(g)})
    # Harmonic_centrality-谐波中心度
    harmonic_c = pd.DataFrame({'Harmonic Centrality': nx.harmonic_centrality(g)})
    # Combining Network Information with Raw Feature Data
    centrality_data = degree_c.join(closeness_c)
    centrality_data = centrality_data.join(betweenness_c)
    centrality_data = centrality_data.join(load_c)
    centrality_data = centrality_data.join(harmonic_c)
    return centrality_data, g, g.pos
