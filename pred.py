import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # 数据加载器
import torchvision.utils as vutils # 用于可视化

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils.ISTD import ISTDDataset
from unet import UNet
from utils.metrics import calculate_psnr, calculate_ssim, MSELoss, RMSELoss


os.chdir('/home/LVM_date2/zjnu/qianlf/deeplearn_Pytorch/Unet_by_labml')
# ----------------------------------------------------------------------------
# pred图片输出路径
output_path = '/home/LVM_date2/zjnu/data/ISTD_output/UNet'
os.makedirs(output_path, exist_ok=True)
# ----------------------------------------------------------------------------
# 读取测试集
# 自定义的数据读取器必须传入Path对象
img_Path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/test/test_A')
mask_Path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/test/test_B')
label_Path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/test/test_C')
# ----------------------------------------------------------------------------
# 设置GPU编号
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------
# 加载模型及训练权重
model = UNet(in_channels=4, out_channels=3)
checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
# ----------------------------------------------------------------------------
# 创建数据集和数据加载器
test_dataset = ISTDDataset(img_Path, mask_Path, label_Path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
# ----------------------------------------------------------------------------
# 预测并保存结果
with torch.no_grad():
    total_psnr = []
    total_ssim = []
    total_mse = []
    total_rmse = []

    for i, data in enumerate(tqdm(test_loader)):
        imgs, targets = data
        
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        # 反归一化后计算评估指标
        outputs = outputs.cpu().clamp(0, 1) * 255.0
        targets = targets.cpu().clamp(0, 1) * 255.0

        # 转换为numpy数组并调整维度 [C, H, W] -> [H, W, C]
        output_img = outputs.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        target_img = targets.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

        output_pil = Image.fromarray(output_img)
        target_pil = Image.fromarray(target_img)

        output_filename = f"pred_{i+1:03d}.png"
        target_filename = f"label_{i+1:03d}.png"
        
        output_pil.save(os.path.join(output_path, output_filename))
        target_pil.save(os.path.join(output_path, target_filename))

        psnr = calculate_psnr(img1=outputs, img2=targets, max_val=255.0)
        ssim = calculate_ssim(img1=outputs, img2=targets, max_val=255.0)
        mse_loss = MSELoss(outputs, targets)
        rmse_loss = RMSELoss(outputs, targets)

        total_psnr.append(psnr)
        total_ssim.append(ssim)
        total_mse.append(mse_loss)
        total_rmse.append(rmse_loss)

    avg_psnr = sum(total_psnr) / len(total_psnr)
    avg_ssim = sum(total_ssim) / len(total_ssim)
    avg_mse_loss = sum(total_mse) / len(total_mse)
    avg_rmse_loss = sum(total_rmse) / len(total_rmse)

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE Loss: {avg_mse_loss:.4f}")
    print(f"Average RMSE Loss: {avg_rmse_loss:.4f}")