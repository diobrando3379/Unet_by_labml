import torch
import torch.nn.functional as F # 用于计算损失函数
import math


# 默认是归一化的, 但是在评估测试集时必须要反归一化到255范围
def calculate_psnr(img1, img2, max_val=1.0):
    """计算PSNR值"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def gaussian_window(size, sigma):
    """创建高斯窗口"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, 1, size) * g.view(1, 1, size, 1)

def calculate_ssim(img1, img2, window_size=11, sigma=1.5, max_val=1.0):
    """计算SSIM值"""
    window = gaussian_window(window_size, sigma).to(img1.device)
    window = window.expand(img1.size(1), 1, window_size, window_size)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean()

def MSELoss(pred, target):
    """计算均方误差损失"""
    return F.mse_loss(pred, target)

def RMSELoss(pred, target):
    """计算均方根误差损失"""
    return torch.sqrt(F.mse_loss(pred, target))