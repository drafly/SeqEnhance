import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from tqdm import tqdm

import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork

from data_loaders.mit5k import mit5k_loader

from model.IAT_main import IAT
from IQA_pytorch import SSIM, MS_SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_test_path', type=str, default="/data/kepler/FiveK/")
config = parser.parse_args()
print(config)

test_dataset = mit5k_loader(root=config.img_test_path, mode='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = IAT().cuda()
model.load_state_dict(torch.load("best_Epoch_mit5k.pth"))
model.eval()


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_test_path + 'Result/'
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(test_loader)):
        
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        enhanced_img = model(low_img)

        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
