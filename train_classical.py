import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
import medmnist
from medmnist import INFO
from tqdm import tqdm

# ================= 配置 =================
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 5
DATA_FLAG = 'pneumoniamnist'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 数据准备 (同上) =================
def get_dataloaders():
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

# ================= 2. 定义纯经典模型 (Classical ViT) =================
class ClassicalViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 原始 ViT head 是 768 -> 1000
        # 我们把量子层去掉，直接用一个线性层代替: 768 -> 2
        self.vit.heads.head = nn.Linear(768, 2)
        
    def forward(self, x):
        return self.vit(x)

# ================= 3. 训练循环 =================
def train():
    train_loader, val_loader = get_dataloaders()
    model = ClassicalViT().to(device) # 使用经典模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.vit.heads.head.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 开始训练 Classical ViT (对照组)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze().long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze().long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"🏁 Epoch {epoch+1} | Train Acc: {100*correct/total:.2f}% | Val Acc: {100*val_correct/val_total:.2f}%")

if __name__ == "__main__":
    train()