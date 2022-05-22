# -*- coding: utf-8 -*-
"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

Use customized CNN to see if image can be classified
to see if the method we used to visualized tabular data can recognize pattern
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 因为要加batchnorm, 所以这里不需要加偏置bias
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=0, bias=False)

        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False)

        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0, bias=False)

        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False)

        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)

        self.cnn6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)

        self.fc1 = nn.Linear(in_features=64 * 9 * 9, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=11)

    def forward(self, x):
        batch = x.size(0)

        out = self.cnn1(x)
        nn.BatchNorm2d(64)
        nn.LeakyReLU(0.2)

        out = self.cnn2(out)
        nn.BatchNorm2d(128)
        nn.LeakyReLU(0.2)

        out = self.cnn3(out)
        nn.BatchNorm2d(256)
        nn.LeakyReLU(0.2)

        out = self.cnn4(out)
        nn.BatchNorm2d(128)
        nn.LeakyReLU(0.2)

        out = self.cnn5(out)
        nn.BatchNorm2d(64)
        nn.LeakyReLU(0.2)

        out = self.cnn6(out)
        nn.BatchNorm2d(64)
        nn.LeakyReLU(0.2)

        # Flatten
        out = out.view(batch, -1)

        out = self.fc1(out)
        nn.LeakyReLU(0.2)
        out = self.fc2(out)
        nn.LeakyReLU(0.2)
        out = self.fc3(out)
        nn.LeakyReLU(0.2)
        out = self.fc4(out)
        nn.LeakyReLU(0.2)
        out = self.fc5(out)
        out = torch.sigmoid(out)

        return out
