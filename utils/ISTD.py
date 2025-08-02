"""
ISTD数据集加载
"""
from pathlib import Path
import torchvision.transforms.functional
from PIL import Image
import torch.utils.data


class ISTDDataset(torch.utils.data.Dataset):
    """
    ## ISTD Dataset
    """

    def __init__(self, image_path: Path, mask_path: Path, label_path: Path):
        """
        image_path: 输入路径
        mask_path: 掩膜路径
        label_path: 标签路径
        """
        # Get a dictionary of images by id
        self.images = {p.stem: p for p in image_path.iterdir()}
        # Get a dictionary of masks by id
        self.masks = {p.stem: p for p in mask_path.iterdir()}
        # Get a dictionary of labels by id
        self.label = {p.stem: p for p in label_path.iterdir()}

        # Image ids list
        self.ids = list(self.images.keys())

        # Transformations
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.ToTensor(),           # RGB 归一化
        ])

    def __getitem__(self, idx: int):
        """
        #### Get an image and its mask.

        :param idx: is index of the image
        """

        # Get image id
        id_ = self.ids[idx]
        # Load image
        image = Image.open(self.images[id_]).convert('RGB') # 根据键名获得对应值(图片路径)
        # Transform image and convert it to a PyTorch tensor
        image = self.transforms(image)

        # Load mask
        mask = Image.open(self.masks[id_]).convert('L')
        # Transform mask and convert it to a PyTorch tensor
        mask = self.transforms(mask)
        m_max = mask.max()
        if m_max > 0:
            mask = mask / m_max
        else:
            mask = torch.zeros_like(mask)

        # Load label
        label = Image.open(self.label[id_]).convert('RGB')
        # Transform label and convert it to a PyTorch tensor
        label = self.transforms(label)

        # 合并图像和掩膜
        images = torch.cat([image, mask], dim=0) # 4通道图像
        # Return the images and the mask
        return images, label

    def __len__(self):
        """
        #### Size of the dataset
        """
        return len(self.ids)

if __name__ == '__main__':
    img_path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/train/train_A')
    mask_path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/train/train_B')
    label_path = Path('/home/Data_Pool/qianlf/ISTD_Dataset/train/train_C')
    ds = ISTDDataset(img_path, mask_path, label_path)
    print(f"Dataset size: {len(ds)}")
    
    # 测试数据加载
    sample_input, sample_target = ds[0]
    print(f"Input shape: {sample_input.shape}")   # 应该是 [4, 512, 512]
    print(f"Target shape: {sample_target.shape}") # 应该是 [3, 512, 512]
    print(f"Input range: [{sample_input.min():.4f}, {sample_input.max():.4f}]")
    print(f"Target range: [{sample_target.min():.4f}, {sample_target.max():.4f}]")