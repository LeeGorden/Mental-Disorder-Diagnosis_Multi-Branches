# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

CV2 draw image to transfer data into image for CV
Compared to Data
"""
import numpy as np
import cv2

class FeaturePixelImg:
    '''
    Create a Pixel image based on the feature
    '''
    def __init__(self, X, epsilon, new_range, line_pixel, channel=3):
        '''
        Function: Create a pixel image for the features
        
        Variable: width-vertical length of the canvas, it is fixed
        
                  line_pixel-horizontal number of pixels assigned to 1 line.
                  ★It better to be a Even number.
                      
                  X-Fetures input as array with no column name
        '''
        self.features = X
        self.epsilon = epsilon
        self.new_range = new_range
        self.norm_std = self._MinMaxNormalization(self.features, self.epsilon, \
                                                  self.new_range)[1]
        
        self.line_pixel = line_pixel
        self.length = 5  *  self.line_pixel
        self.width = 18 * self.line_pixel
        self.channel = channel
        #Create a black canvas
        self.canvas = np.zeros((self.width, self.length, self.channel), \
                               dtype = np.uint8)

    '''---------------------------------------------------------------------'''
    def _RoundArray(self, X):
        '''
        Function: Round an feature Array X with no digit to put in pixel Img
                  since BGR will automatically round float input as int
        '''
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
        '''
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
                 
        ''' 
        try:
            feature_num = X.shape[1]
        except:
            feature_num = 1

        norm_X = np.zeros((len(X), feature_num), dtype = float)
        
        # When test data is input, need to use norm_std of train to standard test
        if test_MinMax == True:
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
        '''
        Function: Draw a line on the canvas
        Variable: start_coordinate-TUPLE of starting pixel(horizontal pixel, 
                  vertal pixel pixel)
                  
                  color-TUPLE of color (B, G, R)
                  
                  line_pixel-The pixels occupied by the line, ★If the starting
                             point is (2, 2), ending point is (2, 4) and the 
                             line_pixel = 2, then the line will be (1~3, 2) to
                             (1~3, 4). In other words, the expended pixels will
                             take the input starting point as It's MIDDLE POINT
        '''
        cv2.fillPoly(self.canvas, points, color)
    
    def _DrawSampleImg(self, sample_color_info, save_dir, Img_name):
        '''
        Function: Draw Image for a sample and save it into a file
        
        Variable: sample-One sample of features,
                  [color info of feature_1, ..., color info of feature_k]
                  color info of feature_1 = [[Blue, color, No], [G], [R]]
                  
                  color_info_length-current data use how many number to express
                  a channel
                  
                  save_dir-location where the image will be saved
                  Img_name-name of image saved
        '''
        try:
            feature_num = self.features.shape[1]
        except:
            feature_num = 1
            
        coordinate_0 = [0, 0]
        coordinate_1 = [self.line_pixel, 0]
        coordinate_2 = [self.line_pixel, self.line_pixel]
        coordinate_3 = [0, self.line_pixel]
        points = np.array([coordinate_0, coordinate_1, coordinate_2, coordinate_3],
                          dtype=np.int32)
        
        for i in range(feature_num):
            G = int(sample_color_info[i])
            self._DrawBlock([points], (0, G, 0))
            if i % 5 == 4:
                coordinate_0[0] = 0
                coordinate_0[1] += self.line_pixel
                coordinate_1[0] = self.line_pixel
                coordinate_1[1] += self.line_pixel
                coordinate_2[0] = self.line_pixel
                coordinate_2[1] += self.line_pixel
                coordinate_3[0] = 0
                coordinate_3[1] += self.line_pixel
            else:
                coordinate_0[0] += int(self.line_pixel)
                coordinate_1[0] += int(self.line_pixel)
                coordinate_2[0] += int(self.line_pixel)
                coordinate_3[0] += int(self.line_pixel)
            points = np.array([coordinate_0, coordinate_1, coordinate_2, coordinate_3],
                          dtype = np.int32)
                
        route = save_dir + '/' + Img_name
        cv2.imwrite(route, self.canvas)
        
    def DrawImg(self, X, save_dir, Img_name, test=False):
        '''
        Function: DrawImg for dataset
        
        Variable: X-Features array
                  
                  save_dir-savingng location
                  
                  Img-names-list of names saving Img
        '''
        if test == False:
            features_normed = self._MinMaxNormalization(X, self.epsilon, self.new_range, test_MinMax=test)[0]
        else:
            features_normed = self._MinMaxNormalization(X, self.epsilon, self.new_range, test_MinMax=test)
        print(features_normed.shape)
        
        for i in range(len(features_normed)):
            self._DrawSampleImg(features_normed[i], save_dir, Img_name[i])


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    data = pd.read_csv('../Data/dataset_cluster.csv')
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1]).reshape(-1, 1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y, random_state=0)
    
    # Create Img name for y_train & y_test for later saving Img as Train/Test_NoOfImg_Label
    ImgName_train = []
    Label_train = []
    for i in range(len(y_train)):
        name = 'Train_' + str(i+1) + '_' + str(y_train[i]) + '.png'
        ImgName_train.append(name)
        Label_train.append(y_train[i][0])  # [0] is needed since y_train is [[0], [1]...]

    ImgName_test = []
    Label_test = []
    for i in range(len(y_test)):
        name = 'Test_' + str(i+1) + '_' + str(y_test[i]) + '.png'
        ImgName_test.append(name)
        Label_test.append(y_test[i][0])

    c = FeaturePixelImg(X_train, epsilon=0.2, new_range=255, line_pixel=4, channel=3)
    # b = c._TransferToColor(X_train)
    # d = c._MinMaxNormalization(X_train, epsilon=0.2, new_range=100)[0]
    # base = c._AssignColor()
    c.DrawImg(X_train, 'C:\Personal\MachineLearning\Python\Algorithm Material\RA\CV\Data2Image\Image\Image_block_5x18\Train', ImgName_train)
    c.DrawImg(X_test, 'C:\Personal\MachineLearning\Python\Algorithm Material\RA\CV\Data2Image\Image\Image_block_5x18\Test', ImgName_test, test=True)
    # e = c._MinMaxNormalization(X_test, epsilon=0.2, new_range=100, test_MinMax=True)
    # f = c._TransferToColor(X_test, cal_num_count=False, test_Transfer=True)
    
    # Create CSV files storing img name for later CNN
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    train_df['file_name'] = ImgName_train
    train_df['class'] = Label_train
    test_df['file_name'] = ImgName_test
    test_df['class'] = Label_test
    train_df.to_csv('C:\Personal\MachineLearning\Python\Algorithm Material\RA\CV\Data2Image\Data\Imginfo_block\Train\Train_Img_Block.csv', index=False)
    test_df.to_csv('C:\Personal\MachineLearning\Python\Algorithm Material\RA\CV\Data2Image\Data\Imginfo_block\Test\Test_Img_Block.csv', index=False)
