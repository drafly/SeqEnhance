# <font color=red>SeqEnhance: A Lightweight Image Processing PipeLine For Low-Light Image Enhancement. </font> 

**2023.5.31:** Upload the low-light image enhancement code.

<br/>

## Abstract

Low-light image enhancement (LLIE) is an important task in computer vision, aiming to improve the visual perception or interpretability of images captured in poorly illuminated environments. Recently, deep learning based methods have been extensively explored to address this issue. While many of these methods have achieved significant advancements in various evaluation metrics for LLIE, only a few have made progress in improving inference speed for the resulting images. As a result, achieving both high-quality enhancements and efficient
inference speed remains a challenge, especially in real-time scenarios. To tackle this challenge, we propose SeqEnhance, a novel lightweight LLIE method based on a predefined parameterized image processing pipeline. Our approach combines the inference capabilities of neural networks for parameter estimation and the efficient processing capabilities of image processing pipelines to generate enhanced images in an end-to-end manner. The experimental results demonstrate that the proposed method achieves competitive performance on image quality evaluation metrics such as PSNR and SSIM with real-time inference speed.

<br/>

## Usage:

### I. Low-Light Enhancement (LOL-V1 dataset, 485 training image, 15 testing image)

1. Download the dataset from the [here](https://daooshee.github.io/BMVC2018website/). The dataset should contains 485 training image and 15 testing image, and should format like:

```
Your_Path
  -- our485
      -- high
      -- low
  -- eval15
      -- high
      -- low
```

2. Evaluation pretrain model on LOL-V1 dataset
```
python evaluation_lol_v1.py --img_val_path Your_Path/eval15/low/
```

Results:
|    | SSIM  | PSNR | enhancement images |
| -- | -- | -- | -- |
|  results  | **0.798**  |  **23.44** | |

3. Training your model on LOL-V1 dataset (get our closely result).

Step 1: crop the LOL-V1 dataset to 256 $\times$ 256 patches:
```
python LOL_patch.py --src_dir Your_Path/our485 --tar_dir Your_Path/our485_patch
```

Step 2: train on LOL-V1 patch images:
```
python train_lol_v1_patch.py --img_path Your_Path/our485_patch/low/ --img_val_path Your_Path/eval15/low/
```

Step 3: tuned the pre-train model (in Step 2) on LOL-V1 patches on the full resolution LOL-V1 image:
```
python train_lol_v1_whole.py --img_path Your_Path/our485/low/ --img_val_path Your_Path/eval15/low/ --pretrain_dir workdirs/snapshots_folder_lol_v1_patch/best_Epoch.pth
```

<br/>


### II. Low-Light Enhancement (MIT-Adobe FiveK _UPE , 4500 training image, 498 testing image)

1. Download the dataset from the [here](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

2. Evaluation pretrain model on FiveK dataset
```
python evaluation_mit5k.py --img_path Your_Path/to/root/dataset/
```

Results:
|    | SSIM  | PSNR | enhancement images |
| -- | -- | -- | -- |
|  results  | **0.894**  |  **24.71** | |

3. Training your model on FiveK dataset. for FiveK, you don't need create patch and directly train is OK. 
   Importantly, our data_loader will resize origin images into 600x450
```
python train_mit5k.py --img_path Your_Path/to/root/dataset/
```

<br/>

