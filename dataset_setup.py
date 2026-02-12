import torch
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义预处理：ViT 通常需要 224x224 的输入，且需要归一化
# PneumoniaMNIST 是灰度图(1通道)，我们需要转成 RGB(3通道) 以适配预训练模型
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # 强制转为3通道
    transforms.Resize((224, 224)),               # 放大到 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloaders(batch_size=32):
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    
    # 自动下载到当前目录
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)

    print(f"数据准备完毕: {data_flag}")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    # 封装为 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_dataloaders()