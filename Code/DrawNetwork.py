# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

CV2 draw image to transfer data into image for CV
Compared to Data
"""
import os
import pandas as pd
import numpy as np
import copy
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import Calculate_Centrality
import Convex_Hull


class FeaturePixelImg:
    """
    Create a Pixel image based on the feature
    """
    def __init__(self, X, epsilon, new_range, line_pixel, channel=3):
        """
        Function: Create a pixel image for the features

        Variable: width-vertical length of the canvas, it is fixed

                  line_pixel-horizontal number of pixels assigned to 1 line.
                  ★It better to be a Even number.

                  X-Fetures input as array with no column name
        """
        self.features = X
        self.epsilon = epsilon
        self.new_range = new_range
        self.norm_std = self._MinMaxNormalization(self.features, self.epsilon, self.new_range)[1]

        self.line_pixel = line_pixel
        self.length = 56 * self.line_pixel  # * max(adjusted_points_range), self.length = self.length.astype(int)
        self.width = 56 * self.line_pixel
        self.channel = channel
        # Create a black canvas
        self.canvas = np.ones((self.width, self.length, self.channel), dtype=np.uint8)
        self.canvas = self.canvas * 255

    '''---------------------------------------------------------------------'''
    def _RoundArray(self, X):
        """
        Function: Round an feature Array X with no digit to put in pixel Img
                  since BGR will automatically round float input as int
        """
        try:
            feature_num = X.shape[1]
        except:
            feature_num = 1

        sample_num = X.shape[0]

        for i in range(feature_num):
            x = X[:, i]
            for j in range(sample_num):
                x[j] = round(x[j])
            X[:, i] = x

        return X

    def _MinMaxNormalization(self, X, epsilon, new_range, test_MinMax=False):
        """
        Function: Control all the features into range >= 0, and into a suitable
                  range to encoding data for BGR img

        Variable: X-Feature Array;

                  Epsilon-cushioning to insure new data >= 0 after same scale
                  MinMaxNormalization, epsilon ∈ [0, 1];

                  new_range-Decide the new range after MinMaxNormalization instead
                  of [0, 1], since pixel will automatically change float into
                  int. In case of distortion, use new_range to decide how detail
                  the number will be. If new_range = 10, new data will be transfer
                  to [0, 10], if = 100, new data will be transfer to [0, 100]

        """
        try:
            feature_num = X.shape[1]
        except:
            feature_num = 1

        norm_X = np.zeros((len(X), feature_num), dtype=float)

        # When test data is input, need to use norm_std of train to standard test
        if test_MinMax:
            norm_std = self.norm_std
            for i in range(feature_num):
                x = X[:, i]
                norm_X[:, i] = (x - norm_std[i][0]) / norm_std[i][1] * new_range
                assert min(norm_X[:, i]) >= 0, 'Need to increase epsilon as cushion'
            return self._RoundArray(norm_X)

        # Create Norm_std to store the key parameters used to normalize data for up comming test data
        norm_std = [(0, 0) for _ in range(feature_num)]
        for i in range(feature_num):
            x = X[:, i]
            # Enlarge the range in case the test data surpass the range of train data
            x_range = (1 + epsilon) * (max(x) - min(x))
            x_min = min(x) - epsilon / 2 * x_range
            x = (x - x_min) / x_range * new_range
            norm_X[:, i] = x
            norm_std[i] = (x_min, x_range)

        return self._RoundArray(norm_X), norm_std

    '''---------------------------------------------------------------------'''
    def _DrawBlock(self, points, color):
        """
        Function: Draw a line on the canvas
        Variable: start_coordinate-TUPLE of starting pixel(horizontal pixel,
                  vertal pixel pixel)

                  color-TUPLE of color (B, G, R)

                  line_pixel-The pixels occupied by the line, ★If the starting
                             point is (2, 2), ending point is (2, 4) and the
                             line_pixel = 2, then the line will be (1~3, 2) to
                             (1~3, 4). In other words, the expended pixels will
                             take the input starting point as It's MIDDLE POINT
        """
        cv2.fillPoly(self.canvas, points, color)

    def _DrawSampleImg(self, sample_color_info, save_dir, Img_name):
        """
        Function: Draw Image for a sample and save it into a file

        Variable: sample-One sample of features,
                  [color info of feature_1, ..., color info of feature_k]
                  color info of feature_1 = [[Blue, color, No], [G], [R]]

                  color_info_length-current data use how many number to express
                  a channel

                  save_dir-location where the image will be saved
                  Img_name-name of image saved
        """
        try:
            feature_num = self.features.shape[1]
        except:
            feature_num = 1

        for i in range(feature_num):
            G = int(sample_color_info[i])
            # 计算像素的四个角左边。 0, 1, 2, 3依次是左上角, 右上角, 右下角, 左下角坐标。
            # 注: 在cv2画图的时候, 横坐标越往右越大和普通直角坐标系一样。但是纵坐标是越往下越大, 和普通坐标系相反, 所以需要调整。
            if i in expand_ind_list:
                coordinate_0 = [adjusted_points_featuremap[i][0] * self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel + 2 * self.line_pixel)]
                coordinate_1 = [adjusted_points_featuremap[i][0] * self.line_pixel + 2 * self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel + 2 * self.line_pixel)]
                coordinate_2 = [adjusted_points_featuremap[i][0] * self.line_pixel + 2 * self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel)]
                coordinate_3 = [adjusted_points_featuremap[i][0] * self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel)]
            else:
                coordinate_0 = [adjusted_points_featuremap[i][0] * self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel + self.line_pixel)]
                coordinate_1 = [adjusted_points_featuremap[i][0] * self.line_pixel + self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel + self.line_pixel)]
                coordinate_2 = [adjusted_points_featuremap[i][0] * self.line_pixel + self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel)]
                coordinate_3 = [adjusted_points_featuremap[i][0] * self.line_pixel,
                                self.width - (adjusted_points_featuremap[i][1] * self.line_pixel)]
            points = np.array([coordinate_0, coordinate_1, coordinate_2, coordinate_3], dtype=np.int32)

            """ 3 color
            if G <= 26:
                self._DrawBlock([points], (255 - G, 0, 0))
            elif G <= 185:
                self._DrawBlock([points], (0, 255 - G, 0))
            else:
                self._DrawBlock([points], (0, 0, 255 - G))
            """
            G = 255 - G
            self._DrawBlock([points], (0, G, 0))

        route = save_dir + '/' + Img_name
        cv2.imwrite(route, self.canvas)

    def DrawImg(self, X, save_dir, Img_name, test=False):
        """
        Function: DrawImg for dataset

        Variable: X-Features array

                  save_dir-savingng location

                  Img-names-list of names saving Img
        """
        if not test:
            features_normed = self._MinMaxNormalization(X, self.epsilon, self.new_range, test_MinMax=test)[0]
        else:
            features_normed = self._MinMaxNormalization(X, self.epsilon, self.new_range, test_MinMax=test)
        print(features_normed.shape)

        for i in range(len(features_normed)):
            self._DrawSampleImg(features_normed[i], save_dir, Img_name[i])


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------------------------
    # Normalize Harmonic Centrality and Importance to find out how many pixel assigned to a feature
    interaction_nodes = Calculate_Centrality.find_interaction(X_train, method='cosine', threshold=0.17)
    centrality, graph_info, nodes_pos_dict = Calculate_Centrality.draw_topology_graph(X_train, interaction_nodes,
                                                                           layout='spring', feature=True)

    # Using SAW to calculate weighted score based on multiple criteria
    for col_names in centrality.columns:
        centrality[[col_names]] = centrality[[col_names]].apply(lambda col:
                                                                (col - np.min(col)) / (np.max(col) - np.min(col)))
    # Set SAW weights as average first, the weights should be hyper-parameter
    centrality_name = list(centrality.columns)
    SAW_weight = [1/len(centrality_name) for _ in range(len(centrality_name))]
    centrality['SAW Score'] = 0
    for row in range(len(centrality)):
        centrality.iloc[row, -1] = np.array(centrality.iloc[row, :-1]).dot(np.array(SAW_weight))
    centrality.sort_values('SAW Score', ascending=False, inplace=True)
    # Filter out SAW Score >= 0.7 and give more pixel on those feature
    list(centrality[centrality['SAW Score'] >= 0.7].index)
    expand_ind_list = [int(feature_name[2:]) - 1
                       for feature_name in list(centrality[centrality['SAW Score'] >= 0.7].index)]

    # Create list of node_pos in an order from C_1 to C_90
    nodes_pos = list()
    for pos in nodes_pos_dict.values():
        nodes_pos.append(pos)
    nodes_pos = np.array(nodes_pos)

    # ----------------------------------制作feature map------------------------------------------------
    # 找到凸包点convex point
    points_data = list()
    for i in range(len(nodes_pos)):
        point_instance = Convex_Hull.Point(nodes_pos[i][0], nodes_pos[i][1])
        points_data.append(point_instance)

    con_points = copy.deepcopy(points_data)
    original_point, con_points = Convex_Hull.get_bottom_point(con_points)
    con_points = Convex_Hull.sort_polar_angle_cos(con_points, original_point)

    con_list = Convex_Hull.graham_scan(con_points, original_point)

    con_array = np.zeros([len(con_list) + 1, 2])
    for i in range(len(con_list)):
        con_array[i, 0] = con_list[i].x
        con_array[i, 1] = con_list[i].y
    con_array[len(con_list)] = [original_point.x, original_point.y]

    plt.figure()
    plt.scatter(nodes_pos[:, 0], nodes_pos[:, 1])
    plt.plot(con_array[:, 0], con_array[:, 1], color='r')
    plt.show()

    # 找到最小外接矩形
    rec_size, rec_location, rec_center = Convex_Hull.find_smallest_rec(con_list)
    rec_location.append(rec_location[0])
    rec_location = np.array(rec_location)

    plt.figure()
    plt.scatter(nodes_pos[:, 0], nodes_pos[:, 1])
    plt.plot(con_array[:, 0], con_array[:, 1], color='r')
    plt.plot(rec_location[:, 0], rec_location[:, 1], color='r')
    plt.scatter(rec_center[0], rec_center[1], color='b')
    plt.show()

    # 将特征点的按照外接矩形进行旋转, 转到外接矩形水平状态没有歪斜
    adjusted_points = Convex_Hull.adjust_rec(points_data, rec_location, rec_center)
    adjusted_points = np.array(adjusted_points)

    plt.figure()
    plt.scatter(adjusted_points[:-5, 0], adjusted_points[:-5, 1], color='b')
    plt.plot(adjusted_points[-5:, 0], adjusted_points[-5:, 1], color='r')
    plt.show()

    # 删除外接矩形顶点坐标并进行normalization, 长宽按照原来的比例, 短边长度变为1
    adjusted_points = adjusted_points[:-5, :]
    adjusted_points_min = np.min(adjusted_points, axis=0)
    adjusted_points_max = np.max(adjusted_points, axis=0)
    adjusted_points_range = (adjusted_points_max - adjusted_points_min) / min(adjusted_points_max - adjusted_points_min)
    adjusted_points_norm = (adjusted_points - adjusted_points_min) / (adjusted_points_max - adjusted_points_min)
    # * adjusted_points_range控制了图片比例, 现在是按照原比例, 若不乘以该值, 表示按照1:1
    adjusted_points_norm = adjusted_points_norm * adjusted_points_range

    plt.figure()
    plt.scatter(adjusted_points_norm[:, 0], adjusted_points_norm[:, 1], color='b')
    plt.show()

    # 调整特征图位置到合适大小, 这里adjust_points_norm * _ 可以调整feature map占用像素长宽
    adjusted_points_featuremap = np.round(adjusted_points_norm * 52, 0)
    adjusted_points_featuremap.astype(int)
    # 返回不重复位置feature的个数
    print(len(np.array(list(set([tuple(t) for t in adjusted_points_featuremap])))))

    plt.figure()
    # plt.scatter(adjusted_points_featuremap[:, 0], adjusted_points_featuremap[:, 1], color='b')
    for i in range(len(adjusted_points_featuremap)):
        x = adjusted_points_featuremap[i, 0]
        y = adjusted_points_featuremap[i, 1]
        plt.scatter(x, y, color='b')
        plt.annotate(features_name[i], xy=(x, y), xytext=(x + 0.2, y + 0.3))
    plt.show()

    # ----------------------------------------------------------------------------------------------------------
    # Create Img name for y_train & y_test for later saving Img as Train/Test_NoOfImg_Label
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    ImgName_train = []
    Label_train = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            class_name = 'Negative'
        else:
            class_name = 'Positive'
        name = class_name + '_Train_' + str(i+1) + '.png'
        ImgName_train.append(name)
        Label_train.append(y_train[i])

    ImgName_test = []
    Label_test = []
    for i in range(len(y_test)):
        if y_test[i] == 0:
            class_name = 'Negative'
        else:
            class_name = 'Positive'
        name = class_name + '_Test_' + str(i+1) + '.png'
        ImgName_test.append(name)
        Label_test.append(y_test[i])

    c = FeaturePixelImg(X_train, epsilon=0.2, new_range=255, line_pixel=4, channel=3)

    c.DrawImg(X_train, os.path.join(project_root,
                                    "Data2Image/Image/Network Science/Image_spring_Hamming_NoInflation_ResNet", "train"),
              ImgName_train)
    c.DrawImg(X_test, os.path.join(project_root,
                                   "Data2Image/Image/Network Science/Image_spring_Hamming_NoInflation_ResNet", "val"),
              ImgName_test, test=True)

    # Create CSV files storing img name for later CNN
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    train_df['file_name'] = ImgName_train
    train_df['class'] = Label_train
    test_df['file_name'] = ImgName_test
    test_df['class'] = Label_test
    train_df.to_csv(os.path.join(project_root, "Data2Image/Data/Imginfo_NetworkScience/Train/Train_Img_NW.csv"),
                    index=False)
    test_df.to_csv(os.path.join(project_root, "Data2Image/Data/Imginfo_NetworkScience/Test/Test_Img_NW.csv"),
                   index=False)
