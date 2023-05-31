import imp
import torch
import torch.nn as nn

from model.light_blocks.cblocks import make_divisible, _conv_bn, SEBlock
from model.light_blocks.MobileNetV2 import InvertedResidualBottleNeck_S


class query_Attention(nn.Module):
    def __init__(self, embed_dim, num_params, num_heads=2, attn_drop=0., bias=True):
        """
        num_params:表示预测的参数总数
        """
        super().__init__()
        self.num_params = num_params
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Parameter(torch.ones((1, self.num_params, embed_dim)), requires_grad=True)
        self.k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_params, C)
        x = self.proj(x)
        return x


class query_SABlock(nn.Module):
    def __init__(self, embed_dim, num_params, ffn_ratio=4., num_heads=2, attn_drop=0.,
                 drop=0., ffn_drop=0.):
        """
        embed_dim:表示token对应的序列长度
        """
        super().__init__()
        attn_unit = query_Attention(
            embed_dim, num_params,
            num_heads=num_heads, attn_drop=attn_drop, bias=True)
        
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=drop)
        )
        
        ffn_hidden_dim = int(embed_dim * ffn_ratio)
        
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=ffn_drop),
            nn.Linear(in_features=ffn_hidden_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        # multi-head attention
        x = x.flatten(2).transpose(1, 2)
        x = self.pre_norm_mha(x)          #N会变成10
        
        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

class Global_pred(nn.Module):
    def __init__(self, in_channels=3, num_heads=4):
        super(Global_pred, self).__init__()
        self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)  
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix
        
        self.generator = query_SABlock(embed_dim=in_channels, num_params=10, num_heads=num_heads)
        
        self.gamma_linear = nn.Linear(in_channels, 1)
        self.color_linear = nn.Linear(in_channels, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.generator(x)
        
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base
        return gamma, color


class Global_pred_Same(nn.Module):
    def __init__(self, in_channels=3, scale=0.5, expand_ratio=6, num_heads=4):
        super(Global_pred_Same, self).__init__()
        self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)  
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix
        
        
         # main blocks
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
            [128, 128, 1]
        ]
        
        conv2 = []

        for i, o, s in depthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2.append(InvertedResidualBottleNeck_S(input_channel, output_channel, 3, 
                                                      stride=s, padding=1, expand_ratio=expand_ratio,
                                                      use_se=True))
            input_channel = output_channel

        self.conv2 = nn.Sequential(*conv2)
        
        self.generator = query_SABlock(embed_dim=output_channel, num_params=10, num_heads=num_heads)
        
        
        self.gamma_linear = nn.Linear(output_channel, 1)
        self.color_linear = nn.Linear(output_channel, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear,)):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.generator(x)
        
        gamma, color = x[:, 0].unsqueeze(1), x[:, 1:]
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        color = self.color_linear(color).squeeze(-1).view(-1, 3, 3) + self.color_base
        return gamma, color
    

class InvertedResidual(nn.Module):
    # MV2block，参考MobileNetv2
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`
    """
 
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                expand_ratio: int = 6, use_se: bool = False) -> None:
        
        assert stride in [1, 2]
        hidden_channels = make_divisible(int(round(in_channels * expand_ratio)), 8)
 
        super(InvertedResidual, self).__init__()
    
        self.use_shortcut = stride == 1 and in_channels == out_channels  #当stride=1且输入特征矩阵与输出特征矩阵shape相同时，会有shortcut连接
        if use_se:
            self.se = SEBlock(out_channels) # SE3
        else:
            self.se = nn.Identity() 
 
        blocks = []
        if expand_ratio != 1:
            # 逐点卷积
            blocks.append(_conv_bn_silu(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, 
                stride=1, padding=0, groups=1))
 
    
        ## DW卷积
        blocks.append(_conv_bn_silu(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, groups=hidden_channels))
        # 逐点卷积
        blocks.append(_conv_bn_silu(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, 
                                    stride=1, padding=0, groups=1, 
                                    use_bn=True, use_act=False))
        # 倒残差
        self.residual = nn.Sequential(*blocks)
 
    def forward(self, x):
        if self.use_shortcut:
            return x + self.se(self.residual(x))
        else:
            return self.se(self.residual(x))
    
    
# 快捷方法，仅为方便
# 用来定义一个普通的nn.Conv2d和一个BN层和一个silu函数
def _conv_bn_silu(in_channels: int, out_channels: int, kernel_size: int,
                  stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, 
                  use_bn: bool = True, use_act: bool = True) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN-SiLu module.
        """
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, 
                                            bias=False))
        if use_bn:
            result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        if use_act:
            result.add_module('act', nn.SiLU())
        return result


if __name__ == "__main__":
    import time

    img = torch.Tensor(8, 3, 400, 600).cuda()
    global_net = Global_pred().cuda()
    
    sum = 0
    for i in range(101):
        start = time.time()
        gamma, color = global_net(img)
        end = time.time()
        print('-------', end - start)
        if i:
            sum += end - start
    print(sum / 100)
    

    