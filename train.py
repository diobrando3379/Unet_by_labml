import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader # 数据加载器
from torch.optim.lr_scheduler import ReduceLROnPlateau # 学习率调度器
from torch.optim import Adam # 优化器
import torch.nn.functional as F # 用于计算损失函数
import torchvision.utils as vutils # 用于可视化
import math
from pathlib import Path
from tqdm import tqdm

from utils.ISTD import ISTDDataset
from unet import UNet

from torch.utils.tensorboard import SummaryWriter

os.chdir('/home/LVM_date2/zjnu/qianlf/deeplearn_Pytorch/Unet_by_labml')

# ----------------------------------------------------------------------------
# 自定义的数据读取器必须传入Path对象
img_Path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/train/train_A')
mask_Path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/train/train_B')
label_Path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/train/train_C')
# ----------------------------------------------------------------------------
# 设置GPU编号
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------
# 超参数
batch_size = 20
epochs = 1000
learning_rate = 1e-4
early_stop = 100

val_rate = 0.2 # 验证集比例
# ----------------------------------------------------------------------------
# 模型保存路径
checkpoints_path = Path('./checkpoints')
checkpoints_path.mkdir(parents=True, exist_ok=True)
# 模型权重保存轮数
save_epoch = 50
# ----------------------------------------------------------------------------
# 读取数据集
dataset = ISTDDataset(img_Path, mask_Path, label_Path)
# 计算训练集和验证集的大小
dataset_size = len(dataset)
val_size = int(dataset_size * val_rate)
train_size = dataset_size - val_size
# 划分训练集和验证集
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)  # 设置随机种子以保证结果可复现
)
train_data_size = len(train_dataset)
test_data_size = len(val_dataset)
print(f"train_data_size:{train_data_size}, test_data_size:{test_data_size}")
# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8, 
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=8, 
    pin_memory=True
)
# ----------------------------------------------------------------------------
# 初始化模型
model = UNet(in_channels=4, out_channels=3)
model = model.to(device)
# 定义损失函数
criterion = nn.MSELoss()
criterion = criterion.to(device)
# 定义优化器
optimizer = Adam(model.parameters(), lr=learning_rate)
# 定义学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
# ----------------------------------------------------------------------------
def calculate_psnr(img1, img2, max_val=1.0):
    """计算PSNR值"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    """计算SSIM值"""
    # 创建高斯窗口
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, 1, size) * g.view(1, 1, size, 1)
    
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
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean()

# ----------------------------------------------------------------------------
total_train_step = 0
total_test_step = 0
best_loss = float('inf')
no_improve_epochs = 0

writer = SummaryWriter('log') # tensorboard

for i in range(epochs):
    print(f"--------epoch:{i+1}--------")

    # 训练开始
    model.train() # 训练模式

    for data in tqdm(train_dataloader):
        imgs, targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, targets)
        
        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"训练次数:{total_train_step}, loss:{loss.item()}")

    # 评估开始
    model.eval() # 评估模式

    total_test_loss = 0
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad(): # 不需要计算梯度
        for data in tqdm(val_dataloader):
            imgs, targets = data
            
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            total_test_loss += loss.item()
            
            # 计算PSNR和SSIM
            psnr = calculate_psnr(outputs, targets)
            ssim = calculate_ssim(outputs, targets)
            
            total_psnr += psnr
            total_ssim += ssim.item()

    scheduler.step(total_test_loss)

    if total_test_loss < best_loss:
        best_loss = total_test_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), checkpoints_path / 'best_model.pth')
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= early_stop:
            print("Early stopping triggered.")
            break

    # 计算平均指标
    avg_psnr = total_psnr / len(val_dataloader)
    avg_ssim = total_ssim / len(val_dataloader)
    avg_test_loss = total_test_loss / len(val_dataloader)
    
    print(f"整体验证集上的 loss:{avg_test_loss}")
    print(f"整体验证集上的 PSNR:{avg_psnr:.4f}")
    print(f"整体验证集上的 SSIM:{avg_ssim:.4f}")
    
    writer.add_scalar('Train/Loss', loss.item(), total_train_step)
    writer.add_scalar('Val/Loss', avg_test_loss, total_test_step)
    writer.add_scalar('Val/PSNR', avg_psnr, total_test_step)
    writer.add_scalar('Val/SSIM', avg_ssim, total_test_step)
    total_test_step += 1

    if (i + 1) % 1 == 0:
        imgs, targets = next(iter(val_dataloader))
        imgs = imgs.to(device)
        targets = targets.to(device)
        model.eval()
        with torch.no_grad():
            preds = model(imgs)
        N = min(4, imgs.size(0))
        pred_show = preds[:N].cpu()
        gt_show = targets[:N].cpu()
        concat = torch.cat([pred_show, gt_show], dim=0)
        grid = vutils.make_grid(concat, nrow=N, normalize=True, scale_each=True)
        writer.add_image('Imgs/Pred_and_Label', grid, global_step=(i + 1))

    # 周期存档
    if (i + 1) % save_epoch == 0:
        torch.save({
            'epoch': i + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }, checkpoints_path / f'checkpoint_epoch_{i+1}.pth')

writer.close()