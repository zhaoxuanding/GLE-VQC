import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
import pennylane as qml
import medmnist
from medmnist import INFO
from tqdm import tqdm
import os

# ================= 配置 (Configuration) =================
# 这是一个经过验证的“黄金配置”，不要轻易改动
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
N_QUBITS = 4
N_LAYERS = 2
DATA_FLAG = 'pneumoniamnist'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "ultimate_hqvit_final.pth"

print(f"🔥 正在启动 Ultimate HQViT (稳定版) | 设备: {DEVICE} | 数据集: {DATA_FLAG}")

# ================= 1. 数据准备 =================
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
    
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), \
           DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= 2. GLE 量子线路 (Method 核心) =================
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def circuit_gle(inputs, weights):
    """
    Global-Local Entangled (GLE) Variational Quantum Circuit
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    
    for i in range(N_LAYERS):
        # 旋转层
        for j in range(N_QUBITS):
            qml.RY(weights[i][j], wires=j)
        
        # 局部环形纠缠
        for j in range(N_QUBITS):
            qml.CNOT(wires=[j, (j + 1) % N_QUBITS])
            
        # 全局长程纠缠 (Small-World Topology)
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
        
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ================= 3. Focal Loss (优化核心) =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = (1 - pt) ** self.gamma * (-log_pt)
        return loss.mean()

# ================= 4. 模型架构 (回归固定缩放) =================
class UltimateHQViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载冻结的 ViT
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads.head = nn.Linear(768, N_QUBITS)
        
        # 定义量子层
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
        self.q_layer = qml.qnn.TorchLayer(circuit_gle, weight_shapes)
        
        self.classifier = nn.Linear(N_QUBITS, 2)
        
    def forward(self, x):
        x = self.vit(x)
        
        # ★★★ 修正回固定值 ★★★
        # 固定乘以 pi，保证数据稳定映射到 [-pi, pi]
        # 这是之前跑出 92.94% 的关键设定，不要改动
        x = torch.tanh(x) * 3.1415926 
        
        x = self.q_layer(x)
        return self.classifier(x)

# ================= 5. 训练流程 =================
def train():
    train_loader, val_loader = get_dataloaders()
    model = UltimateHQViT().to(DEVICE)
    
    # 你的核心竞争力：Focal Loss
    criterion = FocalLoss().to(DEVICE)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    
    print(f"\n🚀 开始训练 (共 {EPOCHS} 轮)...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1}: Val Acc = {acc:.2f}% (Loss: {train_loss/len(train_loader):.4f})")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\n🏆 训练结束! 最高准确率: {best_acc:.2f}%")
    print(f"💾 模型已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    train()