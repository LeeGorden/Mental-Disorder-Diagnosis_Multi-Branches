# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use VAE to see if image can be classified-to see if the method we used to visualized tabular data can recognize pattern
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

latent_dim = 32
inter_dim = 256
mid_dim = (3, 9, 9)  # feature map after last layer in encode, size = 3 * 9 * 9 = C * H * W
mid_num = 1
for i in mid_dim:
    mid_num *= i


class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        """
        Variable: image_size-size of flatten image pixel length
                  z_dim-dimension of each picture in latent space.
                        如果z_dim = 2, 则隐空间用四维向量表示, 前两个是μ, 后两个是σ。每组(μ, σ)表示一个维度上的高斯分布
        """
        super(ConvVAE, self).__init__()


        # 编码器设置
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2),

                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2),

                                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(0.2),

                                     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2),

                                     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2),

                                     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, bias=False),
                                     nn.BatchNorm2d(3),
                                     nn.LeakyReLU(0.2),
                                     # final out put = (3 * 9 * 9)
                                     )

        self.fc1 = nn.Linear(in_features=mid_num, out_features=inter_dim)
        self.fc2 = nn.Linear(in_features=inter_dim, out_features=latent_dim * 2)

        self.fcr2 = nn.Linear(latent_dim, inter_dim)
        self.fcr1 = nn.Linear(inter_dim, mid_num)

        # 解码器设置
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=3, out_channels=64,
                                                        kernel_size=3, stride=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2),

                                     nn.ConvTranspose2d(in_channels=64, out_channels=128,
                                                        kernel_size=3, stride=1, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2),

                                     nn.ConvTranspose2d(in_channels=128, out_channels=256,
                                                        kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(0.2),

                                     nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                                        kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2),

                                     nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                        kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2),

                                     nn.ConvTranspose2d(in_channels=64, out_channels=3,
                                                        kernel_size=4, stride=2, bias=False),
                                     nn.Sigmoid()  # 这里decoder最后要用Sigmoid是因为tensor最后输出各通道内颜色任用0~1表示
                                     )

    # 得到隐空间高斯分布μ, σ后进行采样
    def reparameterize(self, mu, logvar):
        """
        return: 返回基于图片隐空间高斯分布(μ, σ^2), 进行采样以decode
                为了防止梯度无法反向传递以调整(μ, σ), 并借(μ, σ)进一步调整encoder的权重, 我们采样的点要用(μ, σ)来表示
                故先对N(0, 1), 即空间维度z_dim = mu.size()内标准正态分布进行随机采样得到ε, 再通过tensorε * σ + μ的方法返还到z_dim维N(μ, σ^2)。
                这样采样的随机点就能用(μ, σ)表示, 从而不会出现“梯度断裂”
        variable: mu-隐空间高斯分布的均值, mu是一个tensor, mu.size = z_dim
        """
        # 高斯分布: 1/(2Π) * [e^(-(x-μ)^2/2*σ^2)]
        # ★这里logvar = log(σ^2), logvar.mul(0.5).exp() = e^(1/2*log(σ^2)) = e^log((σ^2)^1/2) = e^log(σ) = σ = std
        std = logvar.mul(0.5).exp()  # .mul(0.5)是将logvar这个tensor整体除以2, .exp()再以e为底依次求对应指数
        # 随机生成标准正态分布中的点ε:
        eps = torch.randn(*mu.size())  # ε是z_dim(z_dim = mu.size())维服从N(0, 1)的tensor.
        eps = eps.to(device)  # 需要将另外建立的tensor都放到gpu上才能进行后续点乘, eps生成时默认再cpu上
        mu = mu.to(device)
        std = std.to(device)
        # * mu.size()表示生成mu.size()个数的N(0, 1)分布的ε
        z = mu + std * eps
        return z

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, channel_num, pixel_length, pixel_width]
        :return: the result of our training model
        """
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        batch = x.size(0)  # 每一批含有的样本的个数

        # encoder
        x = self.encoder(x)
        # print(x.size())
        # print(x.view(batch, -1).size())
        x = self.fc1(x.view(batch, -1))
        h = self.fc2(x)
        mu, logvar = torch.chunk(h, 2, dim=1)  # chunk这里表示将h分成2块, 按找列竖着分(dim=1按列, dim=0按行), 分完之后μ,σ各一行表一个样本
        z = self.reparameterize(mu, logvar)  # 采样得到z

        # decoder
        decode = self.fcr2(z)
        decode = self.fcr1(decode)
        # print(decode.size())
        # print(decode.view(batch, *mid_dim).size())
        x_hat = self.decoder(decode.view(batch, *mid_dim))

        return x_hat, mu, logvar
