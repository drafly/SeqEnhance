import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import torchvision
import types
import random

from model.light_blocks.cblocks import _conv_bn, make_divisible
from model.light_blocks.MobileNetV1 import DepthSeparableConv2d, DepthSeparableConv2d_S
from model.light_blocks.MobileNetV2 import InvertedResidualBottleNeck, InvertedResidualBottleNeck_S
from model.light_blocks.MobileNetV3 import Block, hswish, SeModule
from model.light_blocks.BSConv import BSConvU, BSConvS, BSConvS_ModelRegLossMixin, BSConvS_S 
from model.light_blocks.MobileOne import MobileOne
from model.light_blocks.ShuffleNet import ShuffleBlock


#S表示Super
class Piecewise_MNv12_S(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, expand_ratio=6, num_nodes=10):
        super(Piecewise_MNv12_S,self).__init__()
        
        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(InvertedResidualBottleNeck_S(input_channel, output_channel, 3, 
                                                    stride=s, padding=1, expand_ratio=expand_ratio))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(InvertedResidualBottleNeck_S(input_channel, input_channel, 3, 
                                                    stride=1, padding=1, expand_ratio=expand_ratio))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            InvertedResidualBottleNeck_S(input_channel, last_channel, 3, 
                                                    stride=2, padding=1, expand_ratio=expand_ratio),
            InvertedResidualBottleNeck_S(last_channel, last_channel, 3, 
                                                    stride=1, padding=1, expand_ratio=expand_ratio)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result

#S表示Super
class Piecewise_MNv1_S(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, expand_ratio=6, num_nodes=10):
        super(Piecewise_MNv1_S,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(DepthSeparableConv2d_S(input_channel, output_channel, 3, stride=s, padding=1))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(DepthSeparableConv2d_S(input_channel, input_channel, 3, stride=1, padding=1))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            DepthSeparableConv2d_S(input_channel, last_channel, 3, stride=2, padding=1),
            DepthSeparableConv2d_S(last_channel, last_channel, 3, stride=1, padding=1)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result

class Piecewise_MNv3(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, num_nodes=10):
        super(Piecewise_MNv3,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        
        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(576, 1280),
            nn.BatchNorm1d(1280),
            hswish(),
            torch.nn.Linear(1280, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.bneck(x)
        x = self.hs2(self.bn2(self.conv2(x)))
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result


class Piecewise_MNv2(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, num_nodes=10):
        super(Piecewise_MNv2,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network 定义网络
        features = []
        input_channel = make_divisible(32 * scale)
        last_channel = make_divisible(1280 * scale)
        # 第一层的特征层
        features.append(_conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                    stride=2, padding=1))
        features.append(nn.ReLU(inplace=True))
        
        # 倒残差模块参数
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * scale)
            for i in range(n):
                stride = s if i == 0 else 1
                #在模型中加入倒残差模块                
                features.append(InvertedResidualBottleNeck(in_channels=input_channel, out_channels=output_channel, 
                                                           stride=stride, expand_ratio=t))
                input_channel = output_channel
         #最后加入一个卷积块
        features.append(_conv_bn(in_channels=input_channel, out_channels=last_channel, kernel_size=1, padding=0))
        features.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*features)

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
#             torch.nn.PReLU(num_parameters=48, init=0.25),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
    
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x = self.features(x) #获得特征图 
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result



class Piecewise_MNv1(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, num_nodes=10):
        super(Piecewise_MNv1,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(DepthSeparableConv2d(input_channel, output_channel, 3, stride=s, padding=1))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(DepthSeparableConv2d(input_channel, input_channel, 3, stride=1, padding=1))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            DepthSeparableConv2d(input_channel, last_channel, 3, stride=2, padding=1),
            DepthSeparableConv2d(last_channel, last_channel, 3, stride=1, padding=1)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result

##################       BSConv       ##################
class Piecewise_MNv1_BSConvU(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, num_nodes=10, with_bn=True, with_se=False):
        super(Piecewise_MNv1_BSConvU,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(BSConvU(input_channel, output_channel, 3, stride=s, padding=1, with_bn=with_bn, with_se=with_se))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(BSConvU(input_channel, input_channel, 3, stride=1, padding=1, with_bn=with_bn, with_se=with_se))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            BSConvU(input_channel, last_channel, 3, stride=2, padding=1, with_bn=with_bn, with_se=with_se),
            BSConvU(last_channel, last_channel, 3, stride=1, padding=1, with_bn=with_bn, with_se=with_se)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result

class Piecewise_MNv1_BSConvS(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, num_nodes=10, with_bn=True, with_se=False):
        super(Piecewise_MNv1_BSConvS,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(BSConvS_S(input_channel, output_channel, 3, stride=s, padding=1, with_bn=with_bn, with_se=with_se))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(BSConvS_S(input_channel, input_channel, 3, stride=1, padding=1, with_bn=with_bn, with_se=with_se))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            BSConvS_S(input_channel, last_channel, 3, stride=2, padding=1, with_bn=with_bn, with_se=with_se),
            BSConvS_S(last_channel, last_channel, 3, stride=1, padding=1, with_bn=with_bn, with_se=with_se)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result
    
    
##################       MobileOne       ##################
class Piecewise_MNv1_MoblieOne(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, num_nodes=10, 
                 inference_mode=False, with_se=False, num_conv_branches=1):
        super(Piecewise_MNv1_MoblieOne,self).__init__()
 
        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(MobileOne(input_channel, output_channel, 3, stride=s, padding=1, 
                                   inference_mode=inference_mode, with_se=with_se, num_conv_branches=num_conv_branches))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(MobileOne(input_channel, input_channel, 3, stride=1, padding=1, 
                                   inference_mode=inference_mode, with_se=with_se, num_conv_branches=num_conv_branches))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            MobileOne(input_channel, last_channel, 3, stride=2, padding=1, 
                      inference_mode=inference_mode, with_se=with_se, num_conv_branches=num_conv_branches),
            MobileOne(last_channel, last_channel, 3, stride=1, padding=1,
                      inference_mode=inference_mode, with_se=with_se, num_conv_branches=num_conv_branches)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result
    
##################       ShuffleNet       ##################
class Piecewise_MNv12_Shufflev1(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, expand_ratio=6, num_nodes=10, groups=1):
        super(Piecewise_MNv12_Shufflev1,self).__init__()

        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        depthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
        ]

        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(ShuffleBlock(input_channel, output_channel, 3, 
                                      stride=s, padding=1, groups=groups, expand_ratio=expand_ratio))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(ShuffleBlock(input_channel, input_channel, 3, 
                                      stride=1, padding=1, groups=groups, expand_ratio=expand_ratio))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            ShuffleBlock(input_channel, last_channel, 3, 
                         stride=2, padding=1, groups=groups, expand_ratio=expand_ratio),
            ShuffleBlock(last_channel, last_channel, 3, 
                         stride=1, padding=1, groups=groups, expand_ratio=expand_ratio)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        result = applyto + self.emodule(applyto, x)
        return result


class EnhancementModule(torch.nn.Module):
    def __init__(self, parameters_count):
        super(EnhancementModule,self).__init__()
        self.parameters_count = parameters_count

    def forward(self, image, parameters):
        return image


class FunctionBasis(EnhancementModule):
    def __init__(self, basis_dimension):
        super(FunctionBasis,self).__init__(basis_dimension * 3)
        self.bdim = basis_dimension

    def expand(self, x):
        """Bx3xHxW -> Bx3xDxHxW  where D is the dimension of the basis."""
        raise NotImplemented

    def forward(self, image, parameters):
        x = self.expand(image)
        w = parameters.view(parameters.size(0), 3, -1)
        return torch.einsum("bcfij,bcf->bcij", (x, w))


class PiecewiseBasis(FunctionBasis):
    def __init__(self, dim):
        super(PiecewiseBasis,self).__init__(dim)
        nodes = torch.arange(dim).view(1, 1, -1, 1, 1).float()
        self.register_buffer("nodes", nodes)

    def expand(self, x):
        x = x.unsqueeze(2)
        return F.relu(1 - torch.abs((self.bdim - 1) * x - self.nodes))


