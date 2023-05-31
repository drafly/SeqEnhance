import torch
import torch.nn as nn
from typing import Tuple
from .cblocks import _conv_bn, SEBlock

class MobileOne(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1, 
                inference_mode: bool = False, with_se: bool = False, num_conv_branches: int = 1):
        """ Construct MobileOne model.
        :param num_nodes: Number of nodes to setup tone adjust function
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOne, self).__init__()
        
        if with_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity() 
            
         ## DW卷积
        self.depthConv = MobileOneBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=in_channels, 
                                        inference_mode=inference_mode, num_conv_branches=num_conv_branches)
            
        # 逐点卷积
        self.pointConv = MobileOneBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=1, padding=0, groups=1, 
                                        inference_mode=inference_mode, num_conv_branches=num_conv_branches)
        
    def forward(self, x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        x = self.se(x)
        return x

# 定义一个深度卷积模块
class MobileOneBlock(nn.Module):
    
    # 除了最后两个参数，接口与普通的nn.Conv2d保持一致
    # r是过参数化的个数，即文中的k
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 inference_mode: bool = False, num_conv_branches: int = 1):
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param num_conv_branches: Number of linear conv branches. kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
        """
        
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.paddding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        
        
        self.activation = nn.ReLU(inplace=True)
        
        if self.inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, 
                                          stride=self.stride, padding=self.paddding, groups=self.groups, 
                                          bias=True)
        else:
            # 定义中间的过参数化分值
            # 注意分组数等于输入通道数
            self.convs = nn.ModuleList([
                _conv_bn(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, 
                         stride=self.stride, padding=self.paddding, groups=self.groups) for _ in range(self.num_conv_branches)
            ])

            # 定义1×1分支即scale branch，注意分组数
            self.conv_1x1 = None
            if self.kernel_size > 1:
                self.conv_1x1 = _conv_bn(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, 
                                         stride=self.stride, padding=0, groups=self.groups)

            # 定义跳跃连接的BN层
            self.skip_bn = nn.BatchNorm2d(num_features=in_channels) \
                if self.out_channels == self.in_channels and self.stride == 1 else None
    
    def forward(self, x):
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.skip_bn is not None:
            identity_out = self.skip_bn(x)

        # Scale branch output
        scale_out = 0
        if self.conv_1x1 is not None:
            scale_out = self.conv_1x1(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.convs[ix](x)

        return self.activation(out)
    
    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.重参数化训练阶段的多分枝结构以获取一个普通的类似CNN的结构。
        """
        if self.inference_mode: #如果推理模型，则退出
            return
        kernel, bias = self._get_kernel_bias()
        # 只留下第一个分支
        self.reparam_conv = nn.Conv2d(in_channels=self.convs[0].conv.in_channels,
                                      out_channels=self.convs[0].conv.out_channels,
                                      kernel_size=self.convs[0].conv.kernel_size,
                                      stride=self.convs[0].conv.stride,
                                      padding=self.convs[0].conv.padding,
                                      dilation=self.convs[0].conv.dilation,
                                      groups=self.convs[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('convs')
        self.__delattr__('conv_1x1')
        if hasattr(self, 'skip_bn'):
            self.__delattr__('skip_bn')

        self.inference_mode = True
    
    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        获取重参数化后单卷积的权重和偏差
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.conv_1x1 is not None:  #（深度可分离卷积中的）逐点卷积， 如果是重参数化架构，即存在为重参数化推理准备的卷积结构
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.conv_1x1) #得到卷积核的权重和偏置
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.skip_bn is not None:   #如果有跳跃连接
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.skip_bn)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches): #（深度可分离卷积中的）深度卷积
            _kernel, _bias = self._fuse_bn_tensor(self.convs[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        将BN层的算子转化为卷积核的乘积和偏置
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        # 当输入的分支是序列时，记录该分支的卷积核参数、BN的均值、方差、gamma、beta和eps（一个非常小的数）
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # 当输入只有BN层时，添加一个只进行线性映射的3x3卷积核和一个偏置
            # 当输入是恒等变换，获取权重和偏置
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,  # i % input_dim 表示第几组(group), 
                                 self.kernel_size // 2, # self.kernel_size // 2表示中心值
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):   #hasattr用于判断对象是否包含对应的属性
            module.reparameterize()
    return model