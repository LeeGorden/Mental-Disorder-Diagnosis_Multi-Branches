# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use VAE to see if image can be classified-to see if the method we used to visualized tabular data can recognize pattern
"""
import os

from PIL import Image

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from Customized_CNN_224_MultiLabel import CNNModel

# 设置模型运行的设备
torch.__version__
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# ----------------------------------------导入数据与设置图像预处理参数----------------------------------------
# 设置transform参数, transforms.Compose会对读取的图片依次进行Compose(list())中list内的转化
data_transform = {
    "train": transforms.Compose([  # convert('RGB')是为了以防万一原图是RGBA四通道
                                 # transforms.RandomResizedCrop(224),
                                 # transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), ]),
    "val": transforms.Compose([  # transforms.Resize(256),
                               # transforms.CenterCrop(224),
                               transforms.ToTensor(), ])}


# 读取图片函数
def load_image_information(path):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(path).convert('RGB')


# 定义自己数据集的数据读入类
class my_data_set(nn.Module):
    def __init__(self, image_file_path, csv_file_path, transform=None, loader=None):
        super(my_data_set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(csv_file_path, 'r')
        fp.readline()
        image_names_info = []
        labels_info = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line = line.strip('\n')
            information = line.split(',')
            image_names_info.append(information[0])
            # 将标签信息由str类型转换为float类型
            labels_info.append([float(i) for i in information[1:len(information)]])
        self.image_path = image_file_path
        self.images = image_names_info
        self.labels = labels_info
        self.transform = transform
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(self.image_path + '/' + imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 需要将标签转换为float类型，BCELoss只接受float类型
        label = torch.FloatTensor(label)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


# 读取图片
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "Image/t-SNE/Image_t_SNE_V5.1_GaussianCanvas")  # image file path
csv_path = os.path.join(data_root, "Data/Imginfo_t_SNE/V5")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = my_data_set(image_file_path=os.path.join(image_path, "train"),
                            csv_file_path=os.path.join(csv_path, "train", "Train_Img_t_SNE_V5.1_GaussianCanvas.csv"),
                            transform=data_transform["train"],
                            loader=load_image_information)

test_dataset = my_data_set(image_file_path=os.path.join(image_path, "val"),
                           csv_file_path=os.path.join(csv_path, "val", "Test_Img_t_SNE_V5.1_GaussianCanvas.csv"),
                           transform=data_transform["val"],
                           loader=load_image_information)

# 统计以便计算每个epoch的平均准确率
train_num = len(train_dataset)
test_num = len(test_dataset)

# 为数据分batch
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 建立模型
model = CNNModel()
model.to(device)

# Prepare configure of training
epochs = 1

best_acc = 0.0
best_acc_separate_labels = torch.tensor([0 for _ in range(len(train_loader.dataset.labels[0]))]).float()
best_acc_separate_labels = best_acc_separate_labels.to(device)
best_epoch = 0

save_path = './model_multilabels_BCELoss_bnreluoutside_gaussian_canvas.pth'
train_steps = len(train_loader)

# 定义损失函数
# nn.BCEWithLogitsLoss把sigmoid和bce的过程放到一起，网络最后输出层就不需要nn.Sigmoid()了
# 如果再网络最后输出加Sigmoid然后再用nn.BCELoss(), 很有可能因为算力不能反向传播, 而nn.BCEWithLogitsLoss相当于优化后的nn.Sigmoid + nn.BCELoss()
loss_fn = nn.BCELoss()


# 定义多标签分来下准确率的计算
def calculate_prediction(model_predicted, accuracy_th=0.5):
    # 注意这里的model_predicted是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    predicted_result = model_predicted > accuracy_th  # tensor之间比较大小跟array一样
    predicted_result = predicted_result.float()  # 因为label是0, 1的float, 所以这里需要将他们转化为同类型tensor

    return predicted_result


# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# ----------------------------------------训练模型----------------------------------------
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0.0
    # 记录每个标签加总准确率
    train_acc_all_labels = 0.0
    # 记录每个标签的各自准确率, 一共有# 这里标签有len(train_loader.dataset.labels[0])个标签
    train_acc_separate_labels = torch.tensor([0 for _ in range(len(train_loader.dataset.labels[0]))]).float()
    train_acc_separate_labels = train_acc_separate_labels.to(device)
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        # 这里images tensor的维度是([batch size, channel, Image Length, Image, Width])
        # Label tensor的维度是([batch size, Label Length])
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        x_hat = model(images)  # 进行forward操作
        # 将概率输出x_hat转化为标签变量0, 1
        predict_y = calculate_prediction(x_hat, accuracy_th=0.5)  # 由于是多标签分类, 这里的predict_y是一个11维tensor
        # torch.eq(input, other, *, out=None)比较两张量是否相同。 计算一个batch 每个label平均准确率之和
        train_acc_all_labels += torch.eq(predict_y, labels).sum().item() / labels.size()[1]
        # .sum(dim=0)是将输出的label命中率按tensor batch size求和(这里是第一维所以sum(dim=0)
        train_acc_separate_labels += torch.eq(predict_y, labels).float().sum(dim=0)  # 如果不把bool转为float, 会变成逻辑运算
        loss = loss_fn(x_hat, labels)
        # 进行反向传播
        optimizer.zero_grad()  # 将梯度(即梯度跟新方向)初始化为0, 对应d_weights = [0] * n
        loss.backward()  # 求反向传播梯度, 并存储在内存中
        """
        # 使用gradient clipping防止梯度爆炸
        for p in model.parameters():
            print(p.grad.norm())  # 打印所有梯度查看是有梯度爆炸
        nn.utils.clip_grad_norm_(p, 10)  # 10为常用梯度跟新模长限制, utils.clip_grad_norm_最后的下划线表示直接inplace替换
        """
        optimizer.step()  # 跟新所有参数, 对应weights = [weights[k] + alpha * d_weights[k] for k in range(n)]

        # 记录当前step, 也就是当前batch的loss
        train_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # train完一个epoch统计accuracy
    train_accurate_all_labels = train_acc_all_labels / train_num
    train_accurate_separate_labels = train_acc_separate_labels / train_num

    model.eval()
    test_acc_all_labels = 0.0
    test_acc_separate_labels = torch.tensor([0 for _ in range(len(test_loader.dataset.labels[0]))]).float()
    test_acc_separate_labels = test_acc_separate_labels.to(device)
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            x_hat = model(test_images)
            predict_y = calculate_prediction(x_hat, accuracy_th=0.5)
            test_acc_all_labels += torch.eq(predict_y, test_labels).sum().item() / test_labels.size()[1]
            test_acc_separate_labels += torch.eq(predict_y, test_labels).float().sum(dim=0)

            test_bar.desc = "test epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

    # test完一个epoch统计accuracy
    test_accurate_all_labels = test_acc_all_labels / test_num
    test_accurate_separate_labels = test_acc_separate_labels / test_num
    print('[epoch %d] train_loss: %.3f' % (epoch + 1, train_loss / train_steps))
    print('train_accuracy_all_labels:')
    print(train_accurate_all_labels)
    print('train_accuracy_separate_labels:')
    print(train_accurate_separate_labels)
    print('val_accuracy_all_labels:')
    print(test_accurate_all_labels)
    print('val_accuracy_separate_labels:')
    print(test_accurate_separate_labels)

    if test_accurate_all_labels > best_acc:
        best_acc = test_accurate_all_labels
        best_acc_separate_labels = test_accurate_separate_labels
        torch.save(model.state_dict(), save_path)
        print('Model Saved')

print('Finished Training')
print('Best Val_accuracy: %.3f' % best_acc)
print('Best Val_accuracy_separate_labels:')
print(best_acc_separate_labels)
