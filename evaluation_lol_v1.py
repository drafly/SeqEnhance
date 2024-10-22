import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork
from tqdm import tqdm
import time

from model.IAT_main import IAT
from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol_v1_new import lowlight_loader_new


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default='/data/kepler/lol_v1/Test/Low/')
config = parser.parse_args()
print(config)

test_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = IAT().cuda()
model.load_state_dict(torch.load("./checkpoint/best_Epoch_G.pth"))
model.eval()


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('Low', 'Result')
    mkdir(result_path)

sum = 0
with torch.no_grad():
    for i, imgs in tqdm(enumerate(test_loader)):
        #print(i)
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])

        start = time.time()
        enhanced_img = model(low_img)
        end = time.time()
        if i:
            sum += end - start

        if config.save:
            torchvision.utils.save_image(enhanced_img, result_path + str(name) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
    print(sum / (len(test_loader) - 1), 's')


SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
