import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

import math

from cblocks import make_divisible, SEBlock
from mobilevit_config import get_config
from transformer import TransformerEncoder


def mobilevit_xxs(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m
 
def mobilevit_xs(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m
 
def mobilevit_s(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super(MobileViT, self).__init__()
 
        image_channels = 3
        out_channels = 16
 
        self.conv_1 = _conv_bn_silu(in_channels=image_channels, out_channels=out_channels, kernel_size=3, 
                                    stride=2, padding=1)
 
        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])
 
        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = _conv_bn_silu(in_channels=out_channels, out_channels=exp_channels, kernel_size=1, 
                                          stride=1, padding=0)
 
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))
 
        # weight init
        self.apply(self.init_parameters)
 
    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)
 
    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []
 
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
 
            layer = InvertedResidual(in_channels=input_channel, out_channels=output_channels,
                                     stride=stride, expand_ratio=expand_ratio)
            block.append(layer)
            input_channel = output_channels
 
        return nn.Sequential(*block), input_channel
 
    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []
 
        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )
 
            block.append(layer)
            input_channel = cfg.get("out_channels")
 
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads
 
        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))
 
        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim
        ))
 
        return nn.Sequential(*block), input_channel
 
    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass
 
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
 
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x
    

class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit                              #输入到Transformer中token长度
        ffn_dim (int): Dimension of the FFN block                                                   #Transformer Encoder MLP第一个全连接层节点个数
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """
 
    def __init__(
        self,
        in_channels: int,
        # 将输入的[B, H, W, C]变为[B, P, N, d]，这里的transformer_dim就是d
        transformer_dim: int,
        # feed forward network，也就是Transformer Encoder中MSA，模块之后的前馈模块
        ffn_dim: int,
        # Transformer block的堆叠次数
        n_transformer_blocks: int = 2,
        # MSA中每个头的维度
        head_dim: int = 32,
        # Transformer Encoder中MSA内部的Dropout
        attn_dropout: float = 0.0,
        # Transformer Encoder中MSA block里Dropout的概率
        dropout: float = 0.0,
        # feed forward network中MLP内的Dropout概率
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        *args,
        **kwargs
    ) -> None:
        super(MobileViTBlock, self).__init__()
 
        conv_3x3_in = _conv_bn_silu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, 
                                    stride=1, padding=1)
        conv_1x1_in = _conv_bn_silu(in_channels=in_channels, out_channels=transformer_dim, kernel_size=1, 
                                    stride=1, padding=0,
                                    use_bn=False, use_act=False)
        
        conv_1x1_out = _conv_bn_silu(in_channels=transformer_dim, out_channels=in_channels, kernel_size=1, 
                                     stride=1, padding=0)
        
        conv_3x3_out = _conv_bn_silu(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=3, 
                                     stride=1, padding=1)
        
 
        # Local representation模块，包括3x3卷积和1x1升维卷积
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)
 
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
         
        # Global representation模块，包括n个Transformer块
        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)
 
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out
 
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
 
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
 
    # unfolding模块实质上就是将[B, C, H, W]-->[BP, N, C]
    # 就是将原来每个pixel与其他所有pixel做MSA变成了patch_h*patch_w份，每份内部做MSA，分批次输入进Transformer中。
    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h #2, 2
        patch_area = patch_w * patch_h # 4
        batch_size, in_channels, orig_h, orig_w = x.shape
        # 向上取整，若是不能整除，则将feature map尺寸扩大到能整除
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
 
        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            # 若是扩大feature map尺寸，则用双线性插值法
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True
 
        # number of patches along width and height
        # patches的数量
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N
 
        # [B, C, H, W] -> [B * C * num_patch_h, patch_h, num_patch_w, patch_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        # P为patches面积大小，N为patches数量
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)
 
        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }
 
        return x, info_dict
 
    # unfolding模块实质上就是将[BP, N, C]-->[B, C, H, W]
    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )
 
        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
 
        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x
 
    def forward(self, x: Tensor) -> Tensor:
        res = x
 
        fm = self.local_rep(x)
 
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)
 
        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)
 
        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)
 
        fm = self.conv_proj(fm)
 
        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm

    
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
    

if __name__ == '__main__':
    import time
    img = torch.randn(8, 3, 256, 256).cuda()
    
    vit = mobilevit_xxs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit.to(device)

    sum = 0
    for i in range(101):
        start = time.time()
        out = vit(img)
        end = time.time()
        print('-------', end - start)
        if i:
            sum += end - start
    print(sum / 100)

#     vit = mobilevit_xs()
#     out = vit(img)
#     print(out.shape)

#     vit = mobilevit_s()
#     out = vit(img)
#     print(out.shape)