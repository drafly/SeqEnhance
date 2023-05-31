import torch.nn as nn
from .cblocks import make_divisible, _conv_bn, SEBlock
from .MobileNetV3 import SeModule


class DepthSeparableConv2d_S(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                use_se: bool = True):
        super(DepthSeparableConv2d_S, self).__init__()
        if use_se:
            # self.se = SeModule(out_channels)
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity() 
            
        ## DW卷积
        self.depthConv = nn.Sequential(
            #kenrel_size基本上都是3，padding都是1
            _conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=in_channels),
            nn.ReLU(inplace=True)
        )
        # 逐点卷积
        self.pointConv = nn.Sequential(
            _conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                    stride=1, padding=0, groups=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        x = self.se(x)
        return x

#https://zhuanlan.zhihu.com/p/604421856
#深度可分离卷积块，由两部分组成，1.单核dw卷积，2.逐点卷积
class DepthSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1):
        super(DepthSeparableConv2d, self).__init__()
        ## DW卷积
        self.depthConv = nn.Sequential(
            #kenrel_size基本上都是3，padding都是1
            _conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=in_channels),
            nn.ReLU(inplace=True)
        )
        # 逐点卷积
        self.pointConv = nn.Sequential(
            _conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                    stride=1, padding=0, groups=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        return x

    
class MobileNetV1(nn.Module):

    def __init__(self, in_channels=3, scale=1.0, num_classes=0, **kwargs):
        super(MobileNetV1, self).__init__()

        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = ConvBNReLU(in_channels, input_channel, 3, 2, padding=1, bias=False)
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 256, 1],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(DepthSeparableConv2d(input_channel, output_channel, 3, stride=s, padding=1, bias=False))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(DepthSeparableConv2d(input_channel, input_channel, 3, stride=1, padding=1, bias=False))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            DepthSeparableConv2d(input_channel, last_channel, 3, stride=2, padding=1, bias=False),
            DepthSeparableConv2d(last_channel, last_channel, 3, stride=2, padding=1, bias=False)
        )
        # 池化层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 线性分类层
        self.fc = nn.Sequential(
            nn.Linear(last_channel, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        # 将三维图像展平为二维分类特征
        x = torch.flatten(x, 1)
        # print(len(x[0]))
        x = self.fc(x)
        return x