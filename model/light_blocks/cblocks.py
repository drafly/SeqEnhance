import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.downsample = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=False)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, _, _ = inputs.size()
        x = self.downsample(inputs)
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x

# 保证特征层数为8的倍数，输入为v和divisor，v是除数，divisor是被除数，将输出值改造为最接近v的divisor的倍数
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)   # 四舍五入取倍数值
    if new_v < 0.9*v:
        new_v += divisor
    return new_v

# 快捷方法，仅为方便
# 用来定义一个普通的nn.Conv2d和一个BN层
def _conv_bn(in_channels: int, out_channels: int, kernel_size: int,
            stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, groups=groups, bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        return result

    