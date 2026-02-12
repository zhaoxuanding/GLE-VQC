import matplotlib.pyplot as plt
import numpy as np

# ================= 真实实验数据录入 =================

# 1. 你的量子模型 (Ours: Ring-VQC + Focal Loss) - 红色
# 特点：起步快，第5轮就到顶了
epochs_ours = [1, 2, 3, 4, 5]
acc_ours = [86.83, 88.55, 90.27, 90.65, 91.22]

# 2. 经典 ViT (Classical ViT) - 蓝色
# 特点：最终很高，但前期爬坡慢 (第3-5轮明显低于量子)
epochs_vit = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
acc_vit = [86.07, 87.40, 87.98, 88.93, 89.31, 89.89, 90.46, 91.41, 91.22, 91.22]

# 3. 经典 Swin (Classical Swin) - 灰色/绿色
# 特点：一直被压制
epochs_swin = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
acc_swin = [75.57, 83.40, 84.92, 84.73, 85.50, 86.83, 87.02, 87.02, 87.60, 88.17]

# ================= 绘图逻辑 =================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# 画 Swin (灰色虚线，作为垫底参照)
ax.plot(epochs_swin, acc_swin, 'v:', color='grey', linewidth=2, label='Classical Swin (88.17%)')

# 画 Classical ViT (蓝色虚线，作为强力对手)
ax.plot(epochs_vit, acc_vit, 's--', color='#1F77B4', linewidth=2, label='Classical ViT (91.41%)')

# 画 Ours (红色实线，主角)
ax.plot(epochs_ours, acc_ours, 'o-', color='#D62728', linewidth=3, markersize=9, label='Ours (HQViT) (91.22%)')

# 装饰图表
ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Convergence Analysis: Quantum vs. Classical Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_ylim(75, 93)

# 关键标注：在第3轮标注差距 (证明我们收敛快)
# Ours: 90.27 vs ViT: 87.98
ax.annotate('Faster Convergence\n(+2.29%)', 
            xy=(3, 90.27), xycoords='data',
            xytext=(2, 91.5), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#D62728'),
            fontsize=11, color='#D62728', fontweight='bold')

plt.tight_layout()
plt.savefig('all_comparison.png', dpi=300)
print("✅ 终极三合一对比图已生成: all_comparison.png")