# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Explain the result with SHAP to generate disorder np map
"""
import os
import gc
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

import numpy as np
import shap
from PIL import Image
import matplotlib.pyplot as plt

from Customized_CNN_224_MultiLabel import CNNModel

# 清空缓存
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
print(gc.collect())

# 导入训练完的 多标签分类 模型
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
state = torch.load('model_multilabels_BCELoss_bnreluoutside.pth')
model = CNNModel()
model.load_state_dict(state)
model.to(device)
data_transform = transforms.Compose([transforms.ToTensor(), ])


# 导入图片集函数
def prepare_images(path_list, total_image_num=200, background_image_num=50, random_state=0):
    """
    return: list of torch.tensor of background images and images used in SHAP
    variable: path_list-file path to save images
              background_image_num-number of background images for SHAP
              total_image_num-number of total images(including background images) for SHAP
              random_state-random seed used to control random.sample()
    """
    imgs_tensor = torch.zeros([1, 3, 224, 224])
    random.Random(random_state).shuffle(path_list)
    for img_path in path_list[3200:total_image_num]:
        print(img_path)
        img = Image.open(img_path).convert('RGB')
        # 这里img tensor维度是[channel, height, weight]因为batch是1, 所以被忽略了, 所以后面LIME里用的image要用unsqueeze添加上去
        img = data_transform(img)
        # print(img.size())
        # img = img.unsqueeze(dim=0)  # 在最前面加一个维度用来记录batch=1, 以便于后面model做predict和lime解释器的运算
        # print(imgs_for_model.size())
        # pytorch中记录tensor的顺序是[batch, channel, height, weight], 符合shap录入的要求, 不需要permute
        img = img.unsqueeze(dim=0)
        imgs_tensor = torch.cat((imgs_tensor, img), 0)
    return imgs_tensor[1:background_image_num + 1], imgs_tensor[background_image_num + 1:]


root = '../../Image/t-SNE/Image_t_SNE_V5.1/train'
images_path = [os.path.join(root, x) for x in os.listdir(root)]
# Due to calculation time, total_image_num to be comparatively low and create disorder np map batch by batch
background, images = prepare_images(images_path, total_image_num=800, background_image_num=50, random_state=501)
background, images = background.to(device), images.to(device)

e = shap.DeepExplainer(model, background)  # background维度[b, c, h, w]
# shap_value是超出当前类平均概率部分的 概率值 的array的list. list的index按照标签分类来
# 从结果上看, shap_value内的array维度是(batch, channel=3, h, w). 当某一层channel的像素区域对标签分类影响为正, 内对应prob>0, 否则<0
# 根据shapley值计算公式推断: 概率为正表示由于有该区域对当前标签分类有正向影响; 概率为负表示由于没有该区域, 对当前标签分类有负向影响
# ★基于上述逻辑来看, 在分配影响区域时候正负概率区域都要看, 因为都是关键影响因子区域
shap_values = e.shap_values(images)  # images维度[b, c, h, w]
a = np.array(shap_values)
np.save(file="SHAP_canvas_data_800samples_50background_rand501_train_5.npy", arr=a)

"""
# plot
# np.swapaxes进行维度转换, 将tensor对应array维度(batch, channel, h, w)转化为image_plot可以借手的维度(batch, h, w, channel)
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]  # shap_numpy在这里不是像素值而是概率值
test_numpy = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, -1), 1, 2)
# plt.imshow(test_numpy[0])
# shap.image_plot(shap概率值, 图像像素值), 他们的维度都是(batch, h, w, channel)
shap.image_plot(shap_numpy, test_numpy)
"""






