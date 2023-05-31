import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from model.light_blocks.cblocks import _conv_bn, make_divisible
from model.light_blocks.MobileNetV2 import InvertedResidualBottleNeck_S

from model.global_net import Global_pred
# from model.local_net import Local_pred_S
from model.LocalNet.tone_adjust import Local_pred_TA
from model.my_global_net import Global_pred_Same


class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True):
        super(IAT, self).__init__()
#         self.local_net = Local_pred_S(in_dim=in_dim)
        self.local_net = Local_pred_TA()
#         self.local_net = nn.Identity()
    
        self.with_global = with_global
        if self.with_global:
#             self.global_net = Global_pred_Same(in_channels=in_dim, expand_ratio=6)
            self.global_net = Global_pred(in_channels=in_dim)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        img_high = img_low
#         img_high = self.local_net(img_low)

        if not self.with_global:
            return img_high
        
        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
#             img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:]) for i in range(b)], dim=0)
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return img_high



        
if __name__ == "__main__":
    img = torch.Tensor(1, 3, 400, 600)
    net = IAT()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    high = net(img)