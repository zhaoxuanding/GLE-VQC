import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pennylane as qml
import medmnist
from medmnist import INFO
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置 =================
N_QUBITS = 4
N_LAYERS = 2
DATA_FLAG = 'pneumoniamnist'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_PATH = "ultimate_hqvit_final.pth"  # 加载你刚才跑出来的那个权重

print(f"🔥 正在加载模型进行深度评估... | 设备: {DEVICE}")

# ================= 1. 复现模型结构 (必须与训练时完全一致) =================
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def circuit_gle(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    for i in range(N_LAYERS):
        for j in range(N_QUBITS):
            qml.RY(weights[i][j], wires=j)
        for j in range(N_QUBITS):
            qml.CNOT(wires=[j, (j + 1) % N_QUBITS])
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class UltimateHQViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(768, N_QUBITS)
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
        self.q_layer = qml.qnn.TorchLayer(circuit_gle, weight_shapes)
        self.classifier = nn.Linear(N_QUBITS, 2)
        
    def forward(self, x):
        x = self.vit(x)
        x = torch.tanh(x) * 3.1415926 
        x = self.q_layer(x)
        return self.classifier(x)

# ================= 2. 加载数据 =================
def get_dataloader():
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    return DataLoader(val_dataset, batch_size=64, shuffle=False)

# ================= 3. 评估与绘图 =================
def evaluate():
    # 初始化模型并加载权重
    model = UltimateHQViT().to(DEVICE)
    try:
        model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
        print("✅ 权重加载成功！")
    except FileNotFoundError:
        print("❌ 错误：找不到权重文件。请确保 ultimate_hqvit_final.pth 存在。")
        return

    loader = get_dataloader()
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())
            
    # --- A. 计算详细指标 ---
    print("\n📊 Classification Report:")
    # target_names=['Normal', 'Pneumonia'] 对应 PneumoniaMNIST 的 0和1
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'], digits=4)
    print(report)
    
    # --- B. 绘制混淆矩阵 ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix (Ours)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n🎨 混淆矩阵已保存为: confusion_matrix.png")
    
    # 保存指标到 txt 以备论文复制
    with open("final_metrics.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    evaluate()