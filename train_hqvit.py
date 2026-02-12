import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pennylane as qml
from torchvision.models import vit_b_16, ViT_B_16_Weights
import medmnist
from medmnist import INFO
from tqdm import tqdm
from train_hqvit import UltimateHQViT
# ================= 配置区域 =================
BATCH_SIZE = 64          
LEARNING_RATE = 1e-4
EPOCHS = 5               
N_QUBITS = 4             
N_LAYERS = 2             
DATA_FLAG = 'pneumoniamnist' 

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 运行设备: {device} (Python 3.11)")

# ================= 1. 数据准备 =================
def get_dataloaders():
    print(f"\n[1/4] 正在准备 {DATA_FLAG} 数据集...")
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

    print(f"✅ 数据就绪! 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# ================= 2. 定义量子层 =================
# 必须在 @qml.qnode 之前定义设备
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # A. 编码部分
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    
    # B. 变分部分 (Ring-VQC 结构)
    for i in range(N_LAYERS):
        for j in range(N_QUBITS):
            qml.RY(weights[i][j], wires=j)
        
        for j in range(N_QUBITS):
            qml.CNOT(wires=[j, (j + 1) % N_QUBITS])
            
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

# ================= 3. 定义混合模型 (HQViT) =================
class HQViT(nn.Module):
    def __init__(self):
        super().__init__()
        print("[2/4] 正在加载预训练 ViT 模型...")
        
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # 冻结 ViT 主体参数
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 修改 Head 输出 N_QUBITS 个特征
        self.vit.heads.head = nn.Linear(768, N_QUBITS)
        
        # 定义量子层
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # 最终分类器
        self.classifier = nn.Linear(N_QUBITS, 2)
        
    def forward(self, x):
        x = self.vit(x)              
        x = torch.tanh(x) * 3.1415 
        x = self.quantum_layer(x)    
        x = self.classifier(x)       
        return x

# ================= 4. 定义 Focal Loss =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = self.alpha * (1 - pt) ** self.gamma * (-log_pt)
        return loss.mean()

# ================= 5. 训练主循环 =================
def train():
    train_loader, val_loader = get_dataloaders()
    model = HQViT().to(device)
    
    # 损失函数与优化器
    criterion = FocalLoss()
    optimizer = optim.Adam(
        list(model.quantum_layer.parameters()) + 
        list(model.classifier.parameters()) + 
        list(model.vit.heads.head.parameters()),
        lr=LEARNING_RATE
    )
    
    print(f"\n[3/4] 开始训练 (共 {EPOCHS} 轮)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
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
        
        train_acc = 100 * correct / total
        
        # 验证过程
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze().long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"🏁 Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    train()
