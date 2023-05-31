import torch.nn as nn
from .pwise import Piecewise
from .pwise_variants import Piecewise_MNv12_S, Piecewise_MNv1, Piecewise_MNv1_S
from .pwise_variants import Piecewise_MNv1_BSConvU, Piecewise_MNv1_BSConvS
from .pwise_variants import Piecewise_MNv1_MoblieOne
from .pwise_variants import Piecewise_MNv12_Shufflev1
from .filter_estimator import FilterEstimator

class Local_pred_TA(nn.Module):
    def __init__(self):
        super(Local_pred_TA, self).__init__()
        block_p = Piecewise_MNv12_S(num_nodes=10, scale=0.5, expand_ratio=6)
#         block_p = Piecewise_MNv1_BSConvS(num_nodes=10, scale=0.5, with_bn=True, with_se=True)
#         block_p = Piecewise_MNv1_MoblieOne(num_nodes=10, scale=0.5, 
#                                            inference_mode=False, with_se=True, num_conv_branches=5)
        block_fe = FilterEstimator(filter_type='single', kener_size=7, sigma=3)
        
        blocks = [block_p, block_fe]
        
        self.net = nn.Sequential(*blocks)

    def forward(self, img):
        return self.net(img)