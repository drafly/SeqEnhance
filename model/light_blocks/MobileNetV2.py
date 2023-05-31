import torch.nn as nn
from .cblocks import make_divisible, _conv_bn, SEBlock
from .MobileNetV3 import SeModule


#倒残差结构
class InvertedResidualBottleNeck_S(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                expand_ratio: int = 6, use_se: bool = True):
        # expand_ratio代表第一个1x1普通卷积的channel放大倍数 
        super(InvertedResidualBottleNeck_S, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels  #当stride=1且输入特征矩阵与输出特征矩阵shape相同时，会有shortcut连接
        if use_se:
#             self.se = SeModule(out_channels) # SE1
            self.se = SEBlock(out_channels) # SE3
        else:
            self.se = nn.Identity() 
        hidden_channels = in_channels * expand_ratio  #隐层的输入通道数，对应中间depthwise conv的输入通道数
        
        blocks = []
        if expand_ratio > 1:
            # 逐点卷积
            blocks.append(_conv_bn(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, 
                stride=1, padding=0, groups=1))
            blocks.append(nn.ReLU(inplace=True))
            
        ## DW卷积
        blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=hidden_channels))
        blocks.append(nn.ReLU(inplace=True))
        # 逐点卷积
        blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, 
                    stride=1, padding=0, groups=1))
        # 倒残差
        self.residual = nn.Sequential(*blocks)
        
    def forward(self, x):
        if self.use_shortcut:
            return x + self.se(self.residual(x))
        else:
            return self.se(self.residual(x))

#倒残差结构
class InvertedResidualBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1, 
                expand_ratio: int = 6):
        # t代表第一个1x1普通卷积的channel放大倍数 
        super(InvertedResidualBottleNeck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels  #当stride=1且输入特征矩阵与输出特征矩阵shape相同时，会有shortcut连接
        hidden_channels = in_channels * expand_ratio  #隐层的输入通道数，对应中间depthwise conv的输入通道数
        
        blocks = []
        if expand_ratio > 1:
            # 逐点卷积
            blocks.append(_conv_bn(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, 
                stride=1, padding=0, groups=1))
            blocks.append(nn.ReLU(inplace=True))
            
        ## DW卷积
        blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=hidden_channels))
        blocks.append(nn.ReLU(inplace=True))
        # 逐点卷积
        blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, 
                    stride=1, padding=0, groups=1))
        # 倒残差
        self.residual = nn.Sequential(*blocks)
        
    def forward(self, x):
        if self.use_shortcut:
            return x + self.residual(x)
        else:
            return self.residual(x)

class MobileNetV2_Rec(nn.Module):
    def __init__(self, in_channels=3, scale=1.0, round_nearest=8,  num_classes=0, **kwargs):
        super(MobileNetV2_Rec, self).__init__()
        features = []
        # 分类类别数
        self.num_classes = num_classes
        input_channel = make_divisible(32 * scale, round_nearest)
        last_channel = make_divisible(1280 * scale, round_nearest)
        # 第一层的特征层
        features.append(ConvBNReLU(in_channels, input_channel, kernel_size=3, stride=2, padding=1))
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
            output_channel = make_divisible(c * scale, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                #在模型中加入倒残差模块                
                features.append(InvertedResidualBottleNeck(input_channel, output_channel, stride, t=t))
                input_channel = output_channel
         #最后加入一个卷积块
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
         # 分类线形层
        if num_classes > 0:
            self.classifier = nn.Sequential(nn.Dropout(0.2),
                                            nn.Linear(last_channel, num_classes))
        #初始化权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.num_classes > 0:
            x = torch.flatten(x, 1)
            # print(len(x))
            x = self.classifier(x)
        return x