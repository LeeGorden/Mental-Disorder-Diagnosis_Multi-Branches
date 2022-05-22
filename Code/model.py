"""
Author: LiGorden
Email: likehao1006@gmail.com
URL: https://www.linkedin.com/in/kehao-li-06a9a2235/
ResearchGate: https://www.researchgate.net/profile/Gorden-Li

ResNet Transfer Learning
"""
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    Residual Block of Res18 & Res34
    BasicBlock(nn.Module) Inherited class nn.Module
    """

    # If Conv kernel_size in each Conv layers of block are same, expansion = 1, expanding filters_num by 1 time
    # Res18 & Res34 has same kernel_size in each block
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        Variable: in_channel-depth of input feature map
                  out_channel-depth of output feature map
                  downsample-Used to control whether to lower the kernel_size or increase channel_num.
                             downsample发生在虚线结构, 旨在调整捷径上的feature map能和每个Residual Block最后曾卷积层输出的feature maps
                             kernel_size, channel_num一样, 这样才能相加。
                             ★: 无论什么深度的ResNet虚线结构发生在每层Residual Blocks的第一个block, 虚线结构内会发生kernel_size和channel
                             的变化; 而实线结构不会发生变化, 所有实线结构内的stride都是1
                             在实线结构, 捷径上的feature map不需要调整, 因为它和当前Residual Block最后曾卷积层输出的feature maps
                             kernel_size, channel_num一样已经一样。
                             即从当前blocks结构跳入下一个blocks时候, 因为kernal_size减半, 需要通过虚线直连结构进行调整
        """
        # Inherit Class nn.Module __init__()
        # super函数的目的使根据找到Basic的父类, .__init__()表示继承父类的init. 这里父类使class中的nn.Module
        super(BasicBlock, self).__init__()

        # Construct customized __init() for Res18 & Res34
        # First Conv layer, 这里stride = stride是因为实线残差结构和虚线残差结构第一层步长不一样。实线stride=1, 虚线stride=2。
        # 实现结构指同block循环; 虚线结构指从当前block结构循环, 进入到下一个block结构循环, 需要对数据进行降维(减少channel, 所以stride=2)
        # 第一层conv的stride参数由BasicBlock中的__init__stride导入, 控制stride参数控制是否是虚线结构。
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # Since we are going to use BN between Conv and activation, we set no bias to this Conv layer
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # Second Conv layer
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        """
        Function: Forward propagation
        """
        # Shortcut of block
        identity = x  # identity is the shortcut of the block
        if self.downsample is not None:
            # When the block do downsample
            identity = self.downsample(x)

        # Main path of block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Function: Define Residual Block for Res50, Res101 and Res152
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    # The third Conv layer filter_num is 4 times previous filter_num, the expansion = 4
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        """
        Variables: groups & width_per_groups-ResNext的在channel分组卷积, 主要出现于50, 101, 152层的结构中。
                   out_channel-block结构中国3x3的卷积层所用的卷积核个数
                   ResNeXt的好处: https://www.zhihu.com/question/323424817
        """
        super(Bottleneck, self).__init__()

        # width 函数用来调整使用ResNet还是ResNext, 当groups = 32, width_per_group = 4时, 表示采用了ResNeXt结构
        # ★为什么要将组数设置成32:
        # 当使用ResNeXt结构时, 每个block相对于同层数的ResNet, 前两层的卷积核个数是ResNet对应layer的两倍, 最后一层卷积核个数相同
        # 当groups = 32, width_per_group = 4时, width = out_channel * 2, 这是ResNeXt前两层与ResNet前两层的区别。
        width = int(out_channel * (width_per_group / 64.)) * groups

        # 这里第一层Conv的stride无论实线结构还是虚线结构都是1, 所以不用另外设置。
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # ResNeXt只有第二层才会对feature map的channels分组卷积, 实现。
        # 只有是虚线结构, 第二层的stride才会=2, 所以第二层Conv中的stride不能直接赋值1, bottleneck的stride参数控制是否是虚线结构
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            # 当不为None, 对应虚线结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 对应残差结构, Res18 & 34对应BasicBlock, Res50 & 101 & 152对应bottleneck
                 blocks_num,  # list, 对应每块Residual Block对应多少个block循环
                 num_classes=1000,  # 训练集分类个数
                 include_top=True,  # 便于之后在ResNet上搭建更复杂结构
                 groups=1,  # 定义Block内第二层卷积层用到的组数, 当groups != 1而=32时, 表示采用ResNeXt结构
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        #  不管ResNet的任何层数的结构, 初始经过3x3 maxpooling后的初始输入特征channel都是64, 所以这里in_channel直接64
        self.in_channel = 64

        #  当groups = 1时, width_per_group = 64, Block循环块中的第二层Conv是64层, 不分组
        self.groups = groups
        self.width_per_group = width_per_group

        #  所有ResNet/ResNeXt一开始共有的 7x7x64, stride=2卷积层与3x3 maxpooling stride=2池化层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2_x对应一系列结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # Conv3_x对应一系列结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # Conv4_x对应一系列结构
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # Conv5_x对应一系列结构
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 采用自适应平均池化采样AdaptiveAvgPool2d。它与nn.AvgPool2d的区别是它只要你填入想要的每张feature map的输出size就行, channel不变
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 添加full connection层(这里也是输出节点层), 输入节点个数为AvgPool2d后的channel数(因为高和宽已经为1), 输出节点个数为分类数
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 对卷积层进行初权重始化
        for m in self.modules():
            # 判断m是否是一直class nn.Conv2d
            if isinstance(m, nn.Conv2d):
                # 采用kaiming_normal初始化权重。更多初始化方法：https://www.aiuai.cn/aifarm613.html
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        Function: Aggregate of Recurrent Residual Blocks with same structure
                  生成conv2_x ~ conv5_x结构
        Variable: block-BasicBlock/Bottleneck
                  channel-conv2_x ~ conv5_x Residual Block结构中第一层conv的卷积核个数filter_num;
                          conv2_x ~ conv5_x依次为: 64, 128, 256, 512
                  block_num-conv2_x ~ conv5_x各自对应有多少个block
                  stride-控制捷径与每层Residual Blocks第一个Residual Block的kernel_size变化
        """
        downsample = None
        # 设置条件决定什么时候用到下采样(虚线结构: 即跨Residual Blocks的结构)
        if stride != 1 or self.in_channel != channel * block.expansion:
            """
            downsample用来控制捷径、conv2_x到conv3_x转换(也包括conv3_x到conv4_x转换, 依次类推)时channel的升维以及kernel_size的剪裁。
            """
            # make_layer的stride控制着当前conv2_x or ... or conv5_x是否含有虚线结构。conv3_x ~ conv5_x都有虚线结构。当有虚线结构, stride = 2
            # 对于Res18/34, maxpooling后的第一个Residual Block不是虚线结构, 因为深度在conv2_x中始终是64, 捷径特没有改变channel
            # 对于Res50/101/152, maxpooling后第一个Residual Block是虚线结构, 因为捷径需要对接256 channels, 这里downsample用来升维捷径, 将64 深度的输入通过捷径变为256深度的捷径作为残差与Residual Block结果相加

            # 对于Res18/34, 第一层Residual Block内没有捷径需要改变channel与kernel_size, 第二层Residual Blocks开始循环的第一个捷径需要改变channel与kernel_size.
            # 对于Res50/101/152, 每层Residual Blocks循环的第一个捷径都需要改变channel. 但是Kernel_size和Res18/34一样从第二层Residual Blocks开始变化, 因为Maxpooling输出已经是56x56了

            # ResNet/ResNeXt只要不是第一块residual block, 接下来的2, 3, 4, 5块residual block都需要downsample来保证与各自接下来模块对接。
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = list()
        # 每层Residual Block中的第一个Residual Block需要分开写, 因为第一个可能会有下采样downsample
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        # 对于ResNet50/101/152, 第一层blocks的第一个block之后, input_channel变化。ResNet18/34不变, 所以用* block.expansion控制。
        self.in_channel = channel * block.expansion

        # 每层Residual Block中的第一个之后的Residual Block结构都一样, block内不再发生kernel_size变化, 捷径上不存在channel变化。所以直接循环
        for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # * + list或者tuple, 可以直接将内部内容转化成非关键参数。 这里相当于把layers内以list形式存储的blocks结构依次组装到nn.Sequential中
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Variable: x-输入图像
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 进入Conv2_x~Conv5_x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 平均池化下采样
        if self.include_top:
            x = self.avgpool(x)
            # 展平处理
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
