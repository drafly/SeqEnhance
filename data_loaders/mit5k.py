import os
import os.path as osp
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image, ImageOps
import glob
import random
import cv2
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
from glob import glob
#By Ziteng Cui, cui@mi.t.u-tokyo.ac.jp 

random.seed(1143)

class mit5k_loader(data.Dataset):

    def __init__(self, root, mode='train', combined=True):
        self.mode = mode
        self.input_files = list()
        self.expertC_files = list()
        self.resize = True
        self.image_size_w = 600
        self.image_size_h = 450
        
        if self.mode == 'train':
            file = open(os.path.join(root,'train_input.txt'),'r')
            tmp_input_files = sorted(file.readlines())
            
            if combined:
                file = open(os.path.join(root,'train_label.txt'),'r')
                tmp_input_files += sorted(file.readlines())
            
            random.shuffle(tmp_input_files)

        elif self.mode == 'test':
            file = open(os.path.join(root,'test.txt'),'r')
            tmp_input_files = sorted(file.readlines())
        
        for i in range(len(tmp_input_files)):
            self.input_files.append(os.path.join(root, "input", tmp_input_files[i][:-1] + ".jpg"))
            self.expertC_files.append(os.path.join(root,"expertC", tmp_input_files[i][:-1] + ".jpg"))
            
        print("Total examples:", len(self.input_files))

    def FLIP_aug(self, low, high):
        if random.random() > 0.5:
            low = cv2.flip(low, 0)
            high = cv2.flip(high, 0)

        if random.random() > 0.5:
            low = cv2.flip(low, 1)
            high = cv2.flip(high, 1)

        return low, high


    def get_params(self, low):
        self.h, self.w = low.shape[0], low.shape[1] # 900, 1200
        #print(self.h, self.w)
        #self.crop_height = random.randint(self.h / 2, self.h)  # random.randint(self.MinCropHeight, self.MaxCropHeight)
        #self.crop_width = random.randint(self.w / 2, self.w)  # random.randint(self.MinCropWidth,self.MaxCropWidth)
        self.crop_height = self.h / 2 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = self.w / 2 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params(low)
        self.i, self.j = int(self.i), int(self.j)
        #if random.random() > 0.5:
        low = low[self.i: self.i + int(self.crop_height), self.j: self.j + int(self.crop_width)]
        high = high[self.i: self.i + int(self.crop_height), self.j: self.j + int(self.crop_width)]
        return low, high

    def __getitem__(self, index):
        
        im_input = cv2.imread(self.input_files[index % len(self.input_files)], cv2.IMREAD_UNCHANGED)
        im_expert = cv2.imread(self.expertC_files[index % len(self.expertC_files)], cv2.IMREAD_UNCHANGED)
        im_name = os.path.splitext(os.path.split(self.input_files[index % len(self.input_files)])[-1])[0]
        
#         if im_input.shape[0] >= im_input.shape[1]:
#             im_input = cv2.transpose(im_input)
#             im_expert = cv2.transpose(im_expert)
        
#         if self.resize:
#             im_input = cv2.resize(im_input, (self.image_size_w, self.image_size_h))
#             im_expert = cv2.resize(im_expert, (self.image_size_w, self.image_size_h))

        
#         if self.mode == 'train':    #data augmentation
#             data_lowlight, data_highlight = self.FLIP_aug(data_lowlight, data_highlight)
#             #data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)
        #print(data_lowlight.shape)
    
        im_input = (np.asarray(im_input[..., ::-1]) / 255.0)
        im_expert = (np.asarray(im_expert[..., ::-1]) / 255.0)

        im_input = torch.from_numpy(im_input).float()  # float32
        im_expert = torch.from_numpy(im_expert).float()  # float32
        
        return im_input.permute(2, 0, 1), im_expert.permute(2, 0, 1), im_name
#         return {"im_input": im_input.permute(2, 0, 1), "im_expert": im_expert.permute(2, 0, 1), "im_name": im_name}

    def __len__(self):
        return len(self.input_files)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train_path = '/home/czt/DataSets/five5k_dataset/Inputs_jpg'
    test_path = '/home/czt/DataSets/five5k_dataset/UPE_testset/Inputs_jpg'
    test_dataset = adobe5k_loader(train_path, mode='train')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                               pin_memory=True)
    for iteration, imgs in enumerate(test_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
        low_img = imgs[0]
        high_img = imgs[1]
        # visualization(low_img, 'show/low', iteration)
        # visualization(high_img, 'show/high', iteration)