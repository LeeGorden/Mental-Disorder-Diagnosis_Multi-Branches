# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use GMM Algorithm to find out sample interaction/Cluster for t-SNE based feature map method
"""
import os
import pandas as pd
import numpy as np
import t_SNE

from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt

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
samples_name_train = X_train.index

# ------------------------------------------运用GMM进行clustering------------------------------------------------------
# Use EM with GMM to make clusters
# GMM model超参数参考https://blog.csdn.net/jasonzhoujx/article/details/81947663
GMM = mixture.GaussianMixture(n_components=3, covariance_type='full')
gmm = GMM.fit(X_train)
gmm_clusters_train = gmm.predict(X_train).reshape(-1, 1)
gmm_clusters_test = gmm.predict(X_test).reshape(-1, 1)

cluster_data_train = X_train.join(y_train)
cluster_data_train['gmm_clusters'] = gmm_clusters_train
cluster_data_train_sub = cluster_data_train[cluster_data_train['gmm_clusters'] == 1]


# Visualize the clusters of GMM using t-SNE
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
x_tsne_dict_euc, x_tsne_euc = t_SNE.get_tsne_location(X_train, names=samples_name_train, dist_method='euclidean')
# x_tsne_dict_mah, x_tsne_mah = t_SNE.get_tsne_location(X_train, names=samples_name_train, dist_method='mahalanobis')
# x_tsne_dict_cos, x_tsne_cos = t_SNE.get_tsne_location(X_train, names=samples_name_train, dist_method='cosine')

# gmm = GMM.fit(x_tsne_euc)
# gmm_labels_train = gmm.predict(x_tsne_euc)
plt.figure()
plt.scatter(x_tsne_euc[:, 0], x_tsne_euc[:, 1], c=gmm_clusters_train, s=25, cmap='viridis')
plt.show()
plt.figure()
plt.scatter(x_tsne_euc[:, 0], x_tsne_euc[:, 1], c=y_train, s=25, cmap='viridis')
plt.show()

# Use RandomForest to calculate the feature importance for drawing pixel
rf = RandomForestClassifier(n_estimators=500, max_depth=None, oob_score=True, random_state=42)
forest = rf.fit(X_train, y_train)
print('---------Forest Score-----------')
print(forest.score(X_test, y_test))
print('---------Forest oob_score-----------')
print(forest.oob_score_)
# feature imp based on OOB score
importance = forest.feature_importances_
sorted_idx = forest.feature_importances_.argsort()
plt.figure()
plt.barh(data.columns[sorted_idx], forest.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.show()

# Use GBDT to fit the data
gbdt = GradientBoostingClassifier()
gradient_boosting = gbdt.fit(X_train, y_train)
print('---------gradient_boosting Score-----------')
print(gradient_boosting.score(X_test, y_test))

# feature explanatory based on SHAP
explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X_train)[1]  # 这里直接用[1]表示attri对样本判断为class = 1的贡献(有正负)。
np.sum(shap_values[0, :])  # 查看X_train[0]样本各个attri对其class=1贡献的总和。

plt.figure()
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test[0])
# 这里返回的是<IPython.core.display.HTML object>, 因为是jupyter里面的html格式显示的图片
plt.show()

# feature explanatory based on LIME
explainer = LimeTabularExplainer(X_train, feature_names=features_name, class_names=list(set(y_train)))
exp = explainer.explain_instance(X_train[1], gradient_boosting.predict_proba)  # 解释第0个样本的规则
fig = exp.as_pyplot_figure()  # 画图
exp.show_in_notebook(show_table=True, show_all=False)  # 画分析图
# 这里返回的是<IPython.core.display.HTML object>, 因为是jupyter里面的html格式显示的图片

exp = explainer.explain_instance(X_train, gradient_boosting.predict_proba)  # 解释样本的规则
