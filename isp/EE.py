import math
import numbers
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve2d

###########原图 - 高斯模糊图像求边缘###########
# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def rgb2yuv(img):
    img_y = 0.299 * img[:, :, :, 0] + 0.587 * img[:, :, :, 1] + 0.144 * img[:, :, :, 2]
    img_u = - 0.1687 * img[:, :, :, 0] - 0.3313 * img[:, :, :, 1] + 0.5 * img[:, :, :, 2] + 0.5
    img_v = 0.5 * img[:, :, :, 0] - 0.4187 * img[:, :, :, 1] - 0.0813 * img[:, :, :, 2] + 0.5

    return img_y, img_u, img_v

def yuv2rgb(img_y, img_u, img_v):
    b, h, w = img_y.shape[0], img_y.shape[1], img_y.shape[2]
    
    img = torch.zeros(b, h, w, 3).cuda()
    img[:, :, :, 0] = img_y + 1.402 * (img_v - 0.5)
    img[:, :, :, 1] = img_y - 0.34414 * (img_u - 0.5) - 0.71414 * (img_v - 0.5)
    img[:, :, :, 2] = img_y + 1.772 * (img_u - 0.5)
    
    return img


def begin_EE(img, amount):
    '''
    amout 输入范围[0,1], 通过超参数alpha进行放大
    '''
    # 超参数设置
    EE_Para = {}
    EE_Para['input_channels'] = 1
    EE_Para['kernel_size'] = 7
    EE_Para['sigma'] = 10
    EE_Para['mode'] = 'reflect'
    EE_Para['alpha'] = 5

    smoothing = GaussianSmoothing(channels=EE_Para['input_channels'], kernel_size=EE_Para['kernel_size'], 
                                  sigma=EE_Para['sigma']).cuda()
    
    
    img_y, img_u, img_v = rgb2yuv(img)            # RGB (BHW3) to YUV (3BHW)
    y = img_y.unsqueeze(-1).permute(0, 3, 1, 2)   # (BHW) --> (BHWC) --> (BCHW)
#     print("Y Component's shape --->", img_y.shape)
    
    psize = EE_Para['kernel_size'] // 2
    y_pad = F.pad(y, (psize, psize, psize, psize), mode=EE_Para['mode'])      #手动镜像padding消除图片边缘影响
    y_blur = smoothing(y_pad)
    
    y_blur = y_blur.permute(0, 2, 3, 1).squeeze(-1)   # (BCHW) --> (BHWC) --> (BHW)
    gaussian_img = yuv2rgb(y_blur, img_u, img_v)      # YUV (3BHW) to RGB (BHW3)  
    
    img_edge = (img - gaussian_img) * amount.view(-1, 1, 1, 1) * EE_Para['alpha']
    img_enhance = img + img_edge
    img_enhance_ = torch.clip(img_enhance, 1e-8, 1.0)
    
    return img_enhance_

if __name__=="__main__":
    import cv2

    # img = np.random.rand(100, 100, 3)
    img = cv2.imread('linghuan.jpg', 1)
    img = np.array(img).astype(np.float32) / 255
    img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
    print("Input image tensor's shape --->", img_tensor.shape)

    EE_Para = {}
    EE_Para['amount'] = 2
    EE_Para['input_channels'] = 1
    EE_Para['kernel_size'] = 3
    EE_Para['sigma'] = 10
    EE_Para['mode'] = 'reflect'
    
    for amout in range(1, 11):
        print('-----------------------------------------')
        print('Current Amout: ', amout)
        print('Current Image: ', 'linghuan_ee_' + str(amout) + '.png')
        EE_Para['amount'] = amout

        img_out = begin_EE(img_tensor, EE_Para)

        img_out = img_out * 255
        img_out = torch.tensor(img_out.squeeze(0), dtype=torch.uint8)
        cv2.imwrite('linghuan_ee_' + str(amout) + '.png', img_out.cpu().numpy())

