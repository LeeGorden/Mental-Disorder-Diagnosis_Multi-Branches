# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

To 3 bigger clusters
"""
import pandas as pd
import copy

# Import Data
data = pd.read_csv('C:/Personal/MachineLearning/Python/Algorithm Material/RA/CV/Data2Image/Data/dataset.csv')

'''Correlation Matrix'''
# Calculate Correlation Matrix
features = data.iloc[:, :-1]
features_corr = features.corr()
features_name = features.columns

for i in range(len(features_corr)):
    for j in range(len(features_corr)):
        features_corr.iloc[i, j] = abs(features_corr.iloc[i, j])


for i in range(len(features_corr)):
    for j in range(len(features_corr)):
        if i >= j:
            features_corr.iloc[i, j] = 0
    
features_corr_backup = copy.deepcopy(features_corr)

# Create Cluster of Features
clusters = []

while list(features_name):
    print('----------------------------------------------------')
    max_cor = features_corr.stack().max()
    max_ind = features_corr.stack().idxmax()
    print([max_ind[0], max_ind[1]])
    features_corr.loc[max_ind[0], max_ind[1]] = 0
    
    if len(clusters) == 0:
        clusters.append([max_ind[0], max_ind[1]])
        features_name = features_name.drop([max_ind[0], max_ind[1]])
        continue
    
    exist_0 = 0
    exist_1 = 0
    cluster_ind_0 = -1
    cluster_ind_1 = -1
    for cluster_no in range(len(clusters)):
        if max_ind[0] in clusters[cluster_no]:
            exist_0 = 1
            cluster_ind_1 = cluster_no
            
        if max_ind[1] in clusters[cluster_no]:
            exist_1 = 1
            cluster_ind_0 = cluster_no
        cluster_no += 1
    print(exist_0)
    print(exist_1)
    
    if exist_0 == 1 and exist_1 == 1:
        continue
    elif exist_0 == 1 and exist_1 == 0:
        features_name = features_name.drop(max_ind[1])
        clusters[cluster_ind_1].append(max_ind[1])
        print(clusters[cluster_ind_1])
    elif exist_0 == 0 and exist_1 == 1:
        features_name = features_name.drop(max_ind[0])
        clusters[cluster_ind_0].append(max_ind[0])
        print(clusters[cluster_ind_0])     
    elif exist_0 == 0 and exist_1 == 0:
        features_name = features_name.drop([max_ind[0], max_ind[1]])
        clusters.append([max_ind[0], max_ind[1]])

# Calculate the distance between clusters
clusters_dist = {}

for i in range(len(clusters)):
    nearest_cluster = None
    nearest_feature = None
    nearest_correlation = 0
    for feature_name_1 in clusters[i]:
        for j in range(len(clusters)):
            if i == j:
                continue
            for feature_name_2 in clusters[j]:
                cur_correlation = features_corr_backup.loc[feature_name_1, feature_name_2]
                if cur_correlation > nearest_correlation:
                    nearest_cluster = j
                    nearest_feature = feature_name_2
                    nearest_correlation = cur_correlation
    print('----------------------------------------------------------')
    print(i)
    print(nearest_cluster)
    print(nearest_feature)
    print(nearest_correlation)
    clusters_dist[nearest_correlation] = [i, nearest_cluster]

clusters_dist = dict(sorted(clusters_dist.items(), key=lambda x: x[0], reverse=True))

# cluster of clusters
features_clusters = []
for correlation, clusters_ind in clusters_dist.items():
    insert_loc = None
    print('----------------------------------------')
    print(clusters_ind)
    for cluster in clusters_ind:
        print(cluster)
        for i in range(len(features_clusters)):
            print(i)
            print(features_clusters[i])
            if cluster in features_clusters[i]:
                insert_loc = i
                break

    if insert_loc is not None:
        features_clusters[i] += clusters_ind
    else:
        features_clusters.append(clusters_ind)
    print(features_clusters)
# Delete repeated value in features_clusters
for i in range(len(features_clusters)):
    features_clusters[i] = list(set(features_clusters[i]))

# Output Cluster
features_order = []
for clusters_ind_list in features_clusters:
    for cluster_no in clusters_ind_list:
        cur_cluster = clusters[cluster_no]
        for i in range(len(cur_cluster)):
            features_order.append(cur_cluster[i])
features_order.append('Class')
data_cluster = data[features_order]
data_cluster.to_csv('C:/Personal/MachineLearning/Python/Algorithm Material/RA/CV/Data2Image/Data/'
                    'dataset_cluster_block_v2.0.csv', index=0)