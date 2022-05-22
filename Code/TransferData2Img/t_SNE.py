# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use DeepInsight algorithm to transfer data to image
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_tsne_location(x, names, target_dim=2, cal_method='barnes_hut', dist_method='cosine',
                      initialize_method='random', random_state=501, customization=False):
    """
    Function: Return normalized location of features in Cartesian coordinate system based on t-SNE
    Variable: x - np.array, high-dim data needed to be embedded to 2D t-SNE plane
              names - list, it can be either names of feature or names of samples
              target_dim - target output dimension of location in t-SNE plane
              cal_method - default 'barnes_hut'.
                           默认情况下，梯度计算算法使用在O（NlogN）时间内运行的Barnes-Hut近似值。
                           method ='exact’将运行在O（N ^ 2）时间内较慢但精确的算法上。
                           当最近邻的误差需要好于3％时，应该使用精确的算法。
                           但是，确切的方法无法扩展到数百万个示例。0.17新版​​功能：通过Barnes-Hut近似优化方法。
              dist_method - distance of calculating conjoint probability between each point
                          can be those in pdist(), eg{'euclidean', 'cosine', 'mahalanobis', 'correlation'}
              initialize_method - method of pre-processing data
                                can be {'pca', 'random'}
    return: x_norm - 0-1 normalized location in t-SNE plane
    """
    # Transferring to location in tsne 2d plane
    tsne = TSNE(n_components=target_dim, method=cal_method, metric=dist_method,
                init=initialize_method, random_state=random_state) #gorden_customize=customization)
    x_tsne = tsne.fit_transform(x)  # return tuple of location in T-SNE 2D plane
    print("Org data dimension is {}.Embedded data dimension is {}".format(x.shape[-1], x_tsne.shape[-1]))

    # location dictionary
    x_tsne_dict = {}
    for i in range(len(names)):
        x_tsne_dict[names[i]] = x_tsne[i]

    # Plot the point in T-SNE 2D plane
    plt.figure()
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.show(block=True)
    return x_tsne_dict, x_tsne
