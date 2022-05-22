# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Explain the result with LIME to generate disorder np map
"""
import os
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random

from Customized_CNN_224_MultiLabel_Copy import CNNModel

# import in multi-label model trained
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
state = torch.load('model_multilabels_V5.1_2.pth')
model = CNNModel()
model.load_state_dict(state)
data_transform = transforms.Compose([transforms.ToTensor(), ])
model.to(device)


# function preparing images
def prepare_img_fn(path_list, label_no):
    """
    return: imgs_for_LIME读取图片为np.array形式list, 为了适配LIME的input(维度为[height, width, channel]的array)
    variable: path_list-图片的绝对路径
    """
    imgs_for_LIME = []
    for img_path in path_list:
        if img_path[len(root) + 10 + label_no] == '0':
            continue
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        # 这里img tensor维度是[channel, height, weight]因为batch是1, 所以被忽略了, 所以后面LIME里用的image要用unsqueeze添加上去
        img = data_transform(img)
        # print(img.size())
        # img = img.unsqueeze(dim=0)  # 在最前面加一个维度用来记录batch=1, 以便于后面model做predict和lime解释器的运算
        # print(imgs_for_model.size())
        # pytorch中记录tensor的顺序是[batch, channel, height, weight],
        # 需要转化为keras中tensor的顺序[batch, height, weight, channel]才能放进LIME, 因为LIME使用tensorflow设计的
        img_for_LIME = img.permute(1, 2, 0)  # permute更改tensor结构, ()内对应原本tensor的维度index, 内顺序为新的tensor结构
        img_for_LIME = np.array(img_for_LIME)
        img_for_LIME = img_for_LIME.astype(np.double)
        imgs_for_LIME.append(img_for_LIME)
    return imgs_for_LIME


class multi_label_image_explainer:
    """
    image_explainer of LIME for multi_label classifications
    """
    def __init__(self, label_no, model_used,
                 top_labels_viewed=2, hide_color=0, num_samples=1000,  random_seed=0, batch_size=20,
                 positive_only=True, num_features=100000, hide_rest=False):
        # import in model_used to predict in pytorch and label index wish to explain
        self.label_no = label_no  # self.label_no要在self.image_data_for_LIME之前赋值, 因为self.image_data_for_LIME会用到
        self.model_used = model_used
        # configure for explainer.explain_instance
        self.top_labels_viewed = top_labels_viewed
        self.hide_color = hide_color
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.batch_size = batch_size
        # configure for explainer.get_image_and_mask
        self.positive_only = positive_only
        self.num_features = num_features
        self.hide_rest = hide_rest
        # 导入lime_image解释器
        self.explainer = lime_image.LimeImageExplainer()

    def target_label_predict(self, image):
        """
        return: pred_prob(array with size [sample number in a batch, class]
        variable: image-image tensor fit for Pytorch model [batch, height width, channel]
                  label_No-the label classification going to be explained, by default it check the first class = 0
        """
        return np.array(torch.sigmoid(self.model_used(image)))[:, self.label_no]

    def model_predict(self, image):
        """
        这时LIME的classification_fn
        return: predict_prob(np.array(1, 2)), recording probability of the class = 0 and class = 1
        variable: image-image array form fit for LIME [batch, height, width, channel]
        """
        # 行数取决于explainer.explain_instance中num_samples, 是生成样本每个batch内的样本个数batch_size
        prob_array = np.ones([self.batch_size, 2])
        model.eval()
        # 因为导入的是image_data_for_LIME, 是array, 要转化成tensor并且更换维度顺序才能再pytorch模型中跑
        image = torch.FloatTensor(image)
        # 这里不能添加维度, 因为classification_fn中在样本周围采样已经生成了batch维度, 若再unsqueeze模型的输入就会变成[1, batch, h, w, c]
        # image = image.unsqueeze(dim=0)
        # print(image.size())
        image = image.permute(0, 3, 1, 2)
        image.to(device)
        with torch.no_grad():
            # 因为Customized_CNN_224_MultiLabel函数内最后loss是BCEWithLogit(), CNN结构最后就没有sigmoid
            prob_array[:, 1] = self.target_label_predict(image)
        prob_array[:, 0] -= prob_array[:, 1]
        return prob_array

    def explain_result(self, image_array, save_path):
        """
        return: explainable area of an image for chosen label
        variable: image_array-[height, weight, channel] array of 1 image
        """
        explanation = self.explainer.explain_instance(image_array, self.model_predict,
                                                      top_labels=self.top_labels_viewed, hide_color=self.hide_color,
                                                      num_samples=self.num_samples, batch_size=self.batch_size,
                                                      random_seed=self.random_seed)
        print('Explanation Finished!')
        # 返回录入原始样本预测结果的prob, 是一个(1, 2) array对应记录target_class = 0和1的概率
        with torch.no_grad():
            sample_pred_prob = \
                self.target_label_predict(torch.FloatTensor(image_array).unsqueeze(dim=0).permute(0, 3, 1, 2))
        # 因为有可能模型预测错误, 所以引入target_label_ranking输入explanation.top_labels[target_label_ranking]
        if sample_pred_prob > 0.5:
            prediction_status = 'True_Prediction'
            print('True Prediction')
            target_label_ranking = 0  # 当预测正确时, positive_prob = top_label[0]对应概率
        else:
            prediction_status = 'Wrong _Prediction'
            print('Wrong Prediction')
            target_label_ranking = 1  # 当预测错误时, positive_prob = top_label[1]对应概率
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[target_label_ranking],
                                                    positive_only=self.positive_only, num_features=self.num_features,
                                                    hide_rest=self.hide_rest)
        plt.figure()
        plt.title(prediction_status)
        # img以array存储解释区域像素信息(0~1), 维度是(行-h, 列-w, channel). hide_rest=True时, 被隐藏区域自动每个channel设为0.5
        image = mark_boundaries(temp / 2 + 0.5, mask)
        plt.imshow(image)
        plt.savefig(save_path)
        plt.close()
        return image


if __name__ == '__main__':
    root = '../../Image/t-SNE/Image_t_SNE_V5.1/val'
    images_path = [os.path.join(root, x) for x in os.listdir(root)]
    image_num = 300
    explanation_area = np.zeros([224, 224, 10])

    # 对D1-D10进行解释
    for i in range(1, 11):
        # input: 输入要解释的label ind以及对应文件夹路径
        label_to_explained = i
        save_path = '../../Image/Interpretable AI/LIME/V5.1/D' + str(i) + ' Recognization/'
        # 导入图片数据并存储为LIME要求的(h, w, c) array形式
        image_data_for_LIME = prepare_img_fn(images_path, label_to_explained)

        MultiLabel_explainer = multi_label_image_explainer(model_used=model, label_no=label_to_explained, random_seed=0,
                                                           top_labels_viewed=2, hide_color=0,
                                                           num_samples=1000, batch_size=10,
                                                           positive_only=True, num_features=5, hide_rest=True)
        # input: 解释选中的图片
        random.seed(100)
        image_ind_chosen = random.sample(range(len(image_data_for_LIME)), image_num)
        count = 1
        for j in image_ind_chosen:
            print('Processing explanation area of label_' + str(i) + ', now it is the ' + str(count) + ' image')
            image_array = MultiLabel_explainer.explain_result(image_data_for_LIME[j],
                                                              save_path=save_path + str(j) + '.png')
            # 因为该方法每个通道内数值一样, 所以只需要读取第一个channel, image_array变成[224, 224]
            image_array = image_array[:, :, 0]
            # 记录用到的区域
            image_array[image_array == 0.5] = 0  # 这里不会有feature像素用到0.5, 所以可以直接价格0.5变为0
            image_array[image_array != 0] = 1  # 将用到的区域记录为1
            explanation_area[:, :, i - 1] += image_array  # 将对应D1~D10放到10个channel
            count += 1

    # save the high-dimension numpy of disorder map
    np.save(file="canvas_data_1000samples.npy", arr=explanation_area)
