# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use VAE to see if image can be classified-to see if the method we used to visualized tabular data can recognize pattern
"""
import os
import random

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vae_CNN_224 import ConvVAE

# 设置模型运行的设备
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


# ----------------------------------------函数与类----------------------------------------
class ImageData(Dataset):
    def __init__(self, root, transformer=None):
        super(ImageData, self).__init__()
        self.root = root
        self.image_path = [os.path.join(root, x) for x in os.listdir(root)]  # 将每张图片的路径存到self.image_path的list中
        random.shuffle(self.image_path)  # 打乱图片顺序

        if transform is not None:
            self.transformer = transformer

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        return self.transformer(self.image_path[item])


def loss_fn(x_hat, x, mu, logvar, beta=1):
    """
    Calculate the loss. Note that the loss includes two parts.
    return: total loss, BCE and KLD of our model
    """
    REC = F.mse_loss(x_hat, x, size_average=False)  # 因为这里RGB图像虽然是二值(0~1之间)的但是有三个channel
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return REC, KLD, REC + beta * KLD


# ----------------------------------------设置参数----------------------------------------
epochs = 100
batch_size = 32

best_loss = 1e9
best_epoch = 0

valid_losses = []
train_losses = []

# 设置transform参数, transforms.Compose会对读取的图片依次进行Compose(list())中list内的转化
# ★toTensor之后图像各通道颜色从(0-255)转化成(0-1), 这是和np表示图像颜色的差别
transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),  # convert('RGB')是为了以防万一原图是RGBA四通道
                                transforms.ToTensor(), ])
# 读取图片
image_train = ImageData('./Image_t_SNE_V5.3/train', transformer=transform)
image_valid = ImageData('./Image_t_SNE_V5.3/val', transformer=transform)
# 为数据分batch
train_loader = DataLoader(image_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(image_valid, batch_size=batch_size, shuffle=False)

model = ConvVAE()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------------------训练模型----------------------------------------
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0.
    train_num = len(train_loader.dataset)

    for idx, x in enumerate(train_loader):
        # idx表示batch index, x对应batch内数据
        batch = x.size(0)
        x = x.to(device)
        x_hat, mu, logvar = model(x)  # 进行forward操作
        # print(x.size())
        # print(x_hat.size())

        recon, kl, loss = loss_fn(x_hat, x, mu, logvar)
        # print(loss)
        # print(loss / 2)
        # print(loss.item())
        train_loss += loss.item()
        loss = loss / batch
        # 因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和, 所以再每个batch的循环中要将梯度归0
        optimizer.zero_grad()  # 将梯度(即梯度跟新方向)初始化为0, 对应d_weights = [0] * n
        loss.backward()  # 求反向传播梯度
        optimizer.step()  # 跟新所有参数, 对应weights = [weights[k] + alpha * d_weights[k] for k in range(n)]

        if idx % 128 == 0:  # 返回每一个epoch第一个batch和最后一个batch train完之后的loss, 128是batch总数
            print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} in Step {idx}")

    train_losses.append(train_loss / train_num)

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.
    valid_num = len(test_loader.dataset)
    model.eval()
    with torch.no_grad():
        for idx, x in enumerate(test_loader):
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            recon, kl, loss = loss_fn(x_hat, x, mu, logvar)

            valid_loss += loss.item()
            valid_kl += kl.item()
            valid_recon += recon.item()

        valid_losses.append(valid_loss / valid_num)

        print(f"Valid loss {valid_loss / valid_num: .3f} \t Recon {valid_recon / valid_num: .3f} \t KL {valid_kl / valid_num: .3f} in epoch {epoch}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            # model.state_dict()保存网络中的参数, 速度快，占空间少
            torch.save(model.state_dict(), 'best_model_V5.3')
            print("Model saved")
