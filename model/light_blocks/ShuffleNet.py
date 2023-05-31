import torch
import torch.nn as nn
from .cblocks import _conv_bn, SEBlock

#倒残差结构改ShuffleBlock
class ShuffleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                expand_ratio: int = 6, use_se: bool = True):
        # t代表第一个1x1普通卷积的channel放大倍数 
        super(ShuffleBlock, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels  #当stride=1且输入特征矩阵与输出特征矩阵shape相同时，会有shortcut连接
        if use_se:
            self.se = SEBlock(out_channels) # SE3
        else:
            self.se = nn.Identity() 
        hidden_channels = in_channels * expand_ratio  #隐层的输入通道数，对应中间depthwise conv的输入通道数
        
        blocks = []
        if expand_ratio > 1:
            # 逐点卷积
            blocks.append(_conv_bn(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, 
                                   stride=1, padding=0, groups=groups))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(Channel_Shuffle(groups))
            
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

# class ShuffleBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
#                 stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
#                 expand_ratio: int = 6, use_se: bool = True):
#         # t代表第一个1x1普通卷积的channel放大倍数 
#         super(ShuffleBlock, self).__init__()
        
#         if use_se:
#             self.se = SEBlock(out_channels) # SE3
#         else:
#             self.se = nn.Identity() 
#         self.use_shortcut = stride == 1 and in_channels == out_channels  #当stride=1且输入特征矩阵与输出特征矩阵shape相同时，会有shortcut连接
#         hidden_channels = in_channels * expand_ratio  #隐层的输入通道数，对应中间depthwise conv的输入通道数
        
#         blocks = []
#         if expand_ratio > 1:
#             # 逐点卷积
#             blocks.append(_conv_bn(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, 
#                 stride=1, padding=0, groups=groups))
#             blocks.append(nn.ReLU(inplace=True))
#             blocks.append(Channel_Shuffle(groups))
        
            
#         ## DW卷积
#         blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, 
#                     stride=stride, padding=padding, groups=hidden_channels))
#         # 逐点卷积
#         if self.use_shortcut:
#             blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, 
#                                    stride=1, padding=0, groups=groups))
#         else:
#             blocks.append(_conv_bn(in_channels=hidden_channels, out_channels=out_channels - in_channels, kernel_size=1, 
#                                    stride=1, padding=0, groups=groups))
#             self.shortcut = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) \
#                 if stride == 2 else nn.Identity() 
            
            
#         # 倒残差
#         self.residual = nn.Sequential(*blocks)
#         self.activation = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         out = self.residual(x)
#         if self.use_shortcut:
#             x = x + out
#         else:
#             x = self.shortcut(x)
#             x = torch.cat([out, x], 1)
#         return self.se(self.activation(x))

    
class Channel_Shuffle(nn.Module):
    def __init__(self,groups):
        super(Channel_Shuffle, self).__init__()
        self.groups = groups

    def forward(self,x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size,self.groups,channels_per_group,height,width)
        x = x.transpose(1,2).contiguous()
        x = x.view(batch_size,-1,height,width)
        return x
