import math
import torch
import torch.nn as nn
from .cblocks import make_divisible, _conv_bn, SEBlock
from .MobileNetV3 import SeModule

class BSConvU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                with_bn=False, with_se=False):
        super(BSConvU, self).__init__()
        if with_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity() 
        
        # pointwise
        self.pointConv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=1, padding=0, groups=1)
        # batchnorm
        if with_bn:
            self.bn_inner = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn_inner = nn.Identity() 
            
        # depthwise
        self.depthConv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=out_channels)
        
        self.bn_outer = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    
    def forward(self, x):
        x = self.pointConv(x)
        x = self.bn_inner(x)
        x = self.depthConv(x)
        x = self.bn_outer(x)
        x = self.relu(x)
        x = self.se(x)
        return x
    
    
class BSConvS(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                p=0.25, min_mid_channels=4,  with_bn=False):
        super(BSConvS, self).__init__()

        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        
        # pointwise1
        self.pointConv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1,
                                   stride=1, padding=0, groups=1)
        # batchnorm
        if with_bn:
            self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        else:
            self.bn1 = nn.Identity() 
        
        # pointwise2
        self.pointConv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1,
                                   stride=1, padding=0, groups=1)
        # batchnorm
        if with_bn:
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn2 = nn.Identity() 
            
        # depthwise
        self.depthConv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=out_channels)
        
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.pointConv1(x)
        x = self.bn1(x)
        x = self.pointConv2(x)
        x = self.bn2(x)
        x = self.depthConv(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x
        
    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")
    
class BSConvS_S(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                p=0.25, min_mid_channels=4,  with_bn=False, with_se=True):
        super(BSConvS_S, self).__init__()
        
        if with_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity() 

        self.bsConvS = BSConvS(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, 
                               with_bn=with_bn)
        
    def forward(self, x):
        x = self.bsConvS(x)
        x = self.se(x)
        return x
        


class BSConvS_ModelRegLossMixin():
    def reg_loss(self, alpha=0.1):
        loss = 0.0
        for sub_module in self.modules():
            if hasattr(sub_module, "_reg_loss"):
                loss += sub_module._reg_loss()
        return alpha * loss