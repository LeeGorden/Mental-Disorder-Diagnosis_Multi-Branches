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
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

import t_SNE
import Convex_Hull
import Calculate_Centrality


class FeaturePixelImg:
    """
    Create a Pixel image based on the feature
    """
    def __init__(self, X, epsilon, new_range, line_pixel, channel=3, pre_canvas=None):
        """
        Function: Create a pixel image for the features
        
        Variable: width-vertical length of the canvas, it is fixed
        
                  line_pixel-horizontal number of pixels assigned to 1 line.
                  ★It better to be a Even number.
                      
                  X-Fetures input as array with no column name

                  pre_canvas-预先准备好的canvas, 若没有则为None, 否则就是一个(h, w, c)的np.array
        """
        self.features = X
        self.epsilon = epsilon
        self.new_range = new_range
        self.norm_std = self._MinMaxNormalization(self.features, self.epsilon, self.new_range)[1]
        
        self.line_pixel = line_pixel
        self.length = 28 * self.line_pixel  # * max(adjusted_points_range), self.length = self.length.astype(int)
        self.width = 28 * self.line_pixel
        self.channel = channel
        # Create a black canvas
        if pre_canvas is not None:
            print('Using pre_canvas')
            self.canvas = pre_canvas
        else:
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
        Function: Draw a block on the canvas
        Variable: start_coordinate-TUPLE of starting pixel(horizontal pixel, 
                  vertical pixel pixel)
                  
                  color-TUPLE of color (B, G, R)
                  
                  line_pixel-The pixels occupied by the line, ★If the starting
                             point is (2, 2), ending point is (2, 4) and the 
                             line_pixel = 2, then the line will be (1~3, 2) to
                             (1~3, 4). In other words, the expended pixels will
                             take the input starting point as It's MIDDLE POINT
        """
        cv2.fillPoly(self.canvas, points, color)

    def _DrawCircle(self, center_x, center_y, radius_size, color):
        """
        Function: Draw a circle on the canvas
        Variable: start_coordinate-TUPLE of starting pixel(horizontal pixel,
                  vertical pixel pixel)

                  color-TUPLE of color (B, G, R)

                  line_pixel-The pixels occupied by the line, ★If the starting
                             point is (2, 2), ending point is (2, 4) and the
                             line_pixel = 2, then the line will be (1~3, 2) to
                             (1~3, 4). In other words, the expended pixels will
                             take the input starting point as It's MIDDLE POINT
        """
        cv2.circle(self.canvas, (center_x, center_y), radius=radius_size, color=color, thickness=-1)

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
            if i in focus_ind_list:
                center_x = adjusted_points_featuremap[i][0] * self.line_pixel + self.line_pixel / 2
                center_y = self.width - (adjusted_points_featuremap[i][1] * self.line_pixel + self.line_pixel / 2 + 3)  # 3用于调整位置
                radius_used = self.line_pixel / 2
                G = 255 - G
                self._DrawCircle(int(center_x), int(center_y), int(radius_used), (G, G, G))

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
                G = 255 - G
                self._DrawBlock([points], (G, G, G))

            """ 3 color
            if G <= 26:
                self._DrawBlock([points], (255 - G, 0, 0))
            elif G <= 185:
                self._DrawBlock([points], (0, 255 - G, 0))
            else:
                self._DrawBlock([points], (0, 0, 255 - G))
            """

            """
            # for multicolor feature pixel
            # CV中颜色顺序为(B, G, R)
            if i in black_list:
                self._DrawBlock([points], (G, G, G))
            elif i in D2_list:  # yellow
                # self._DrawBlock([points], (G, 255, 255))
                self._DrawBlock([points], (G, 255, G))
            elif i in D1_list:  # sky-blue
                # self._DrawBlock([points], (255, 255, G))
                self._DrawBlock([points], (G, 255, G))
            elif i in D4_list:  # blue
                self._DrawBlock([points], (255, G, G))
            elif i in D6_list:  # pink
                # self._DrawBlock([points], (255, G, 255))
                self._DrawBlock([points], (G, G, 255))
            elif i in D3_list:  # red
                # self._DrawBlock([points], (G, G, 255))
                self._DrawBlock([points], (255, G, G))
            elif i in D5_list:  # blue
                self._DrawBlock([points], (255, G, G))
            elif i in D7_list:  # pink
                # self._DrawBlock([points], (255, G, 255))
                self._DrawBlock([points], (G, G, 255))
            elif i in D8_list:  # Green
                # self._DrawBlock([points], (G, 255, G))
                self._DrawBlock([points], (G, G, 255))
            elif i in D9_list:  # blue
                # self._DrawBlock([points], (255, G, G))
                self._DrawBlock([points], (G, 255, G))
            elif i in D10_list:  # yellow
                # self._DrawBlock([points], (G, 255, 255))
                self._DrawBlock([points], (G, G, 255))
            """
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
    data = pd.read_csv(data_root + '/multilabel.csv')
    features = list(data.columns[:-11])
    X, y = np.array(data)[:, :-11], np.array(data)[:, -11]
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y, random_state=0)
    # 因为要把features而不是samples用t-SNE可视化, 所以要对X_train进行转置
    X_tsne = X_train.T
    x_tsne_dict, x_tsne = t_SNE.get_tsne_location(X_tsne, names=features, initialize_method='random',
                                                  dist_method='cosine', cal_method='barnes_hut', random_state=501,
                                                  customization=False)  # customization是自行改写源码的部分, 对应gorden_customize
    # ----------------------------------制作feature map------------------------------------------------
    # 找到凸包点convex point
    points_data = list()
    for i in range(len(x_tsne)):
        point_instance = Convex_Hull.Point(x_tsne[i][0], x_tsne[i][1])
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
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.plot(con_array[:, 0], con_array[:, 1], color='r')
    plt.show(block=True)

    # 找到最小外接矩形
    rec_size, rec_location, rec_center = Convex_Hull.find_smallest_rec(con_list)
    rec_location.append(rec_location[0])
    rec_location = np.array(rec_location)

    plt.figure()
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.plot(con_array[:, 0], con_array[:, 1], color='r')
    plt.plot(rec_location[:, 0], rec_location[:, 1], color='r')
    plt.scatter(rec_center[0], rec_center[1], color='b')
    plt.show(block=True)

    # 将特征点的按照外接矩形进行旋转, 转到外接矩形水平状态没有歪斜
    adjusted_points = Convex_Hull.adjust_rec(points_data, rec_location, rec_center)
    adjusted_points = np.array(adjusted_points)

    plt.figure()
    plt.scatter(adjusted_points[:-5, 0], adjusted_points[:-5, 1], color='b')
    plt.plot(adjusted_points[-5:, 0], adjusted_points[-5:, 1], color='r')
    plt.show(block=True)

    # 删除外接矩形顶点坐标并进行normalization, 长宽按照原来的比例, 短边长度变为1
    adjusted_points = adjusted_points[:-5, :]
    adjusted_points_min = np.min(adjusted_points, axis=0)
    adjusted_points_max = np.max(adjusted_points, axis=0)
    adjusted_points_range = (adjusted_points_max - adjusted_points_min) / min(adjusted_points_max - adjusted_points_min)
    adjusted_points_norm = (adjusted_points - adjusted_points_min) / (adjusted_points_max - adjusted_points_min)
    # * adjusted_points_range控制了图片比例, 现在是按照原比例, 若不乘以该值, 表示按照1:1
    # adjusted_points_norm = adjusted_points_norm * adjusted_points_range

    plt.figure()
    plt.scatter(adjusted_points_norm[:, 0], adjusted_points_norm[:, 1], color='b')
    plt.show(block=True)

    # 调整特征图位置到合适大小, 这里adjust_points_norm * _ 可以调整feature map占用像素长宽
    adjusted_points_featuremap = np.round(adjusted_points_norm * 27, 0)
    adjusted_points_featuremap.astype(int)
    # 返回不重复位置feature的个数
    print(len(np.array(list(set([tuple(t) for t in adjusted_points_featuremap])))))

    plt.figure()
    plt.scatter(adjusted_points_featuremap[:, 0], adjusted_points_featuremap[:, 1], color='b')
    for i in range(len(adjusted_points_featuremap)):
        x = adjusted_points_featuremap[i, 0]
        y = adjusted_points_featuremap[i, 1]
        plt.scatter(x, y, color='b')
        plt.annotate(features[i], xy=(x, y), xytext=(x + 0.2, y + 0.3))
    # plt.imshow(gaussian_canvas)
    plt.show(block=True)
    np.save("feature_map_location.npy", adjusted_points_featuremap)
    # ---------------------------------------对数据运用ML进行初步训练---------------------------------------------
    # 划分数据集, 因为最后按照location作图和获取location用的可能不是同一个training set(取决于不同研究目的下label的划分)
    adjusted_points_featuremap = np.load("feature_map_location.npy")
    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../.."))
    data_root = os.path.join(project_root, "Data", "Original Data")
    data = pd.read_csv(data_root + '/multilabel.csv')
    features = list(data.columns[-10:])
    X = np.array(data.iloc[:, :-11])
    y = np.array(data.iloc[:, -11:])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y[:, 0], random_state=0)
    D_data = y_train.T
    D_cosine_dist = squareform(pdist(D_data, metric='cosine'))
    D_data = y_train[:, 1:].T
    D_tsne_dict, D_tsne = t_SNE.get_tsne_location(D_data, names=features, initialize_method='random',
                                                  dist_method='cosine', cal_method='barnes_hut', random_state=501,
                                                  customization=False)
    plt.figure()
    plt.scatter(D_tsne[:, 0], D_tsne[:, 1], color='b')
    for i in range(len(D_tsne)):
        x = D_tsne[i, 0]
        y = D_tsne[i, 1]
        plt.scatter(x, y, color='b')
        plt.annotate(features[i], xy=(x, y), xytext=(x + 0.2, y + 0.3))
    plt.show(block=True)
    # ----------------------------------------------------------------------------------------------------------

    focus_ind_list = []
    """
    focus_ind_list = [43, 65, 69, 8, 41, 57, 9, 82,
                      17, 31, 70, 28, 54, 52, 48, 58, 77, 53, 27,
                      60, 10, 33, 29, 39, 2, 78, 0, 81, 23, 80,
                      88, 49, 22, 32, 73, 89, 66, 74, 79, 86, 44, 59, 63]
    """
    """
    # 根据LIME解释区域划分色块
    black_list = [76, 50, 28, 54, 31, 2, 30, 56, 70, 18]  # 中间与多中心里疾病相关的点
    D2_list = [87, 68, 44, 74, 85, 79, 86, 37, 35, 33, 10, 45, 78]  # D2
    D3_list = [24, 43, 69, 7, 42, 36, 60, 6, 5]
    D4_list = [23, 1, 73, 27, 53, 29]
    D5_list = [0, 39, 12, 49, 22, 46, 13, 15, 77]
    D7_list = [72, 81, 71, 32, 65]
    D8_list = [67, 75, 82, 17, 9]
    D9_list = [20, 8, 34, 25, 40, 4, 21, 64]
    D10_list = [19, 84, 88]
    D1_list = [26, 55, 57, 41, 51, 52, 48, 16, 14, 47, 3, 58, 38, 11]  # D1
    D6_list = [62, 66, 80, 83, 59, 63, 61, 89]  # D6
    """
    # ----------------------------------------------------------------------------------------------------------
    # Create Img name for y_train & y_test for later saving Img as Train/Test_NoOfImg_Label
    ImgName_train = []
    Label_train = []
    for i in range(len(y_train)):
        if y_train[i][0] == 0:
            class_name = 'Negative'
        else:
            class_name = 'Positive'
        name = class_name + '_'
        for j in range(len(y_train[i])):
            name += str(y_train[i][j])
        name = name + '_Train_' + str(i+1) + '.png'
        ImgName_train.append(name)
        Label_train.append(y_train[i])

    ImgName_test = []
    Label_test = []
    for i in range(len(y_test)):
        if y_test[i][0] == 0:
            class_name = 'Negative'
        else:
            class_name = 'Positive'
        name = class_name + '_'
        for j in range(len(y_test[i])):
            name += str(y_test[i][j])
        name = name + '_Test_' + str(i+1) + '.png'
        ImgName_test.append(name)
        Label_test.append(y_test[i])

    gaussian_canvas = np.load('Gaussian_canvas_sigma1.npy')
    c = FeaturePixelImg(X_train, epsilon=0.05, new_range=255, line_pixel=8, channel=3, pre_canvas=gaussian_canvas)
    # b = c._TransferToColor(X_train)
    # d = c._MinMaxNormalization(X_train, epsilon=0.2, new_range=100)[0]
    # base = c._AssignColor()
    c.DrawImg(X_train, os.path.join(project_root, "Data2Image/Image/t-SNE/Image_t_SNE_Final", "train"),
              ImgName_train)
    c.DrawImg(X_test, os.path.join(project_root, "Data2Image/Image/t-SNE/Image_t_SNE_Final", "val"),
              ImgName_test, test=True)
    # e = c._MinMaxNormalization(X_test, epsilon=0.2, new_range=100, test_MinMax=True)
    # f = c._TransferToColor(X_test, cal_num_count=False, test_Transfer=True)
    
    # Create CSV files storing img name for later CNN
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    train_df['Image_name'] = ImgName_train
    train_df = pd.concat([train_df, pd.DataFrame(columns=['class', 'D_1', 'D_2', 'D_3', 'D_4', 'D_5',
                                                          'D_6', 'D_7', 'D_8', 'D_9', 'D_10'],
                                                 data=Label_train)], axis=1)

    test_df['Image_name'] = ImgName_test
    test_df = pd.concat([test_df, pd.DataFrame(columns=['class', 'D_1', 'D_2', 'D_3', 'D_4', 'D_5',
                                                        'D_6', 'D_7', 'D_8', 'D_9', 'D_10'],
                                               data=Label_test)], axis=1)
    train_df.to_csv(os.path.join(project_root,
                                 "Data2Image/Data/Imginfo_t_SNE/V5/train/Train_Img_t_SNE_Final.csv"),
                    index=False)
    test_df.to_csv(os.path.join(project_root,
                                "Data2Image/Data/Imginfo_t_SNE/V5/val/Test_Img_t_SNE_Final.csv"),
                   index=False)

    """
    # field experiment
    field_experiment_tabular = pd.DataFrame(X_test)
    field_experiment_tabular = pd.concat([field_experiment_tabular,
                                         pd.DataFrame(columns=
                                                      ['class', 'D_1', 'D_2', 'D_3', 'D_4', 'D_5',
                                                       'D_6', 'D_7', 'D_8', 'D_9', 'D_10'], data=y_test)], axis=1)
    field_experiment_tabular.to_csv('d.csv')
    """

