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
# 为了公平对比，所有参数必须和你的量子版保持一致！
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10  # 跑10轮，为了画对比图
DATA_FLAG = 'pneumoniamnist'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据准备 =================
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

# ================= 纯经典 ViT 模型 =================
class ClassicalViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载完全一样的预训练权重
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # 冻结参数 (和量子版保持一致，这样才公平)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 原始 ViT head 是 768 -> 1000
        # 我们直接接一个线性层: 768 -> 2
        # 注意：这里没有量子层，所以参数量比量子版稍微多一点点（或持平），
        # 我们要证明“加了量子层效果更好”或者“收敛更快”。
        self.vit.heads.head = nn.Linear(768, 2)
        
    def forward(self, x):
        return self.vit(x)

# ================= 训练循环 =================
def train():
    train_loader, val_loader = get_dataloaders()
    model = ClassicalViT().to(device)
    criterion = nn.CrossEntropyLoss() # 经典模型通常就用 CE Loss
    optimizer = optim.Adam(model.vit.heads.head.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🚀 开始训练 Classical ViT (基准对照组)...")
    
    # 用于记录数据，回头画图用
    acc_history = []
    loss_history = []

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
        
        val_acc = 100 * val_correct / val_total
        acc_history.append(val_acc)
        loss_history.append(train_loss / len(train_loader))
        
        print(f"🏁 Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")

    print("\n✅ 训练结束！请复制下面的数据用于画图：")
    print(f"classical_acc = {acc_history}")
    print(f"classical_loss = {loss_history}")

if __name__ == "__main__":
    train()