import torch
import torch.nn as nn
import torchvision
import torch.optim
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
import types
import wandb 

import os
import argparse
import numpy as np
from tqdm import tqdm

from data_loaders.lol_v1_whole import lowlight_loader_new
from model.IAT_main import IAT

from IQA_pytorch import SSIM
from utils import PSNR, validation, get_mean_ssim_and_psnr

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--img_path', type=str, default='/data/kepler/lol_v1/Train/Low/')
parser.add_argument('--img_val_path', type=str, default='/data/kepler/lol_v1/Test/Low/')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--pretrain_dir', type=str, default='./checkpoint/best_Epoch_G.pth')

parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--warmup_epochs', type=int, default=20)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="/output/snapshots_folder_lol_v1_whole")

config = parser.parse_args()
print(config)

#可视化
os.environ['WANDB_API_KEY'] = '754184bd8b3fdf261d10e61b401c8e7c5904f120'
wandb.init(project="IAT_Ablation")
wandb.config = {
  "learning_rate": config.lr,
  "epochs": config.num_epochs,
  "batch_size": config.batch_size
}

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Model Setting
model = IAT().cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting
train_dataset = lowlight_loader_new(images_path=config.img_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

num_step = int(config.num_epochs * len(train_loader))
warmup_step = int(config.warmup_epochs * len(train_loader))
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=config.weight_decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_step)
scheduler = CosineLRScheduler(optimizer, t_initial=num_step, lr_min=0.0, 
                              warmup_lr_init=0.0, warmup_t=warmup_step, t_in_epochs=False)



device = next(model.parameters()).device
print('the device is:', device)

# wandb.watch() automatically fetches all layer dimensions, gradients, model parameters
# and logs them automatically to your dashboard.
# using log="all" log histograms of parameter values in addition to gradients
wandb.watch(model, log="all")

# Loss & Optimizer Setting & Metric
L1_loss = nn.L1Loss()


ssim = SSIM()
psnr = PSNR()
psnr_high = 0

model.train()
print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    epoch_loss, epoch_psnr, epoch_ssim, iters = 0, 0, 0, 0
    
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()

        optimizer.zero_grad()
        model.train()
        enhance_img = model(low_img)

        loss = L1_loss(enhance_img, high_img)
        
        #训练过程计算PSNR和SSIM
        ssim_mean_train, psrn_mean_train = get_mean_ssim_and_psnr(enhance_img, high_img)
        epoch_loss += loss.item()
        epoch_psnr += psrn_mean_train
        epoch_ssim += ssim_mean_train
        iters += 1
        
        loss.backward()
        optimizer.step()
        # scheduler.step()
        scheduler.step_update(epoch * len(train_loader) + iteration)

        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

    # Evaluation Model
    model.eval()
    ssim_mean, psnr_mean = validation(model, val_loader)
    
    wandb.log({
            "Epoch_Loss": epoch_loss / iters,
            "Epoch_psnr": epoch_psnr / iters,
            "Epoch_ssim": epoch_ssim / iters,
            "SSIM_mean_test": ssim_mean,
            "PSNR_mean_test": psnr_mean
            })

    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(ssim_mean) + 'the PSNR is' + str(psnr_mean) + '\n')

    if psnr_mean > psnr_high:
        psnr_high = psnr_mean
        print('the highest SSIM value is:', str(psnr_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))

    f.close()
