import matplotlib.pyplot as plt

# ================= 真实实验数据 =================
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 基准模型 (Paper Fig.6)
acc_paper = [86.26, 88.36, 89.31, 89.70, 90.08, 91.03, 91.03, 91.79, 91.98, 92.37]

# 我们的模型 (Ours GLE)
acc_ours = [86.64, 87.60, 89.69, 90.27, 91.60, 91.22, 92.18, 92.37, 92.75, 92.94]

# ================= 绘图代码 =================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# 画 Baseline (蓝色虚线)
ax.plot(epochs, acc_paper, 's--', color='#1F77B4', linewidth=2, 
        alpha=0.8, label='Baseline (Standard HQViT)')

# 画 Ours (红色实线，加粗)
ax.plot(epochs, acc_ours, 'o-', color='#D62728', linewidth=3, 
        markersize=8, label='Ours (GLE-VQC + Focal)')

# --- 装饰图表 ---
ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Standard vs. Enhanced HQViT', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
ax.grid(True, linestyle='--', alpha=0.6)

# 设置Y轴范围：给顶部留出空间，防止标注超出边界
ax.set_ylim(min(acc_paper)-1, max(acc_ours)+1.8) 
# 设置X轴刻度：确保1-10完整显示
ax.set_xticks(epochs)

# --- 标注1: 最终提升 (调整了 xytext 避免重合) ---
final_gap = acc_ours[-1] - acc_paper[-1]
ax.annotate(f'+{final_gap:.2f}% Improvement', 
            xy=(10, acc_ours[-1]), 
            xycoords='data',
            xytext=(10.2, 93.4), # 向右上方移动，脱离密集区
            textcoords='data',
            ha='right',          # 水平对齐
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='#D62728', lw=1.5),
            fontsize=12, color='#D62728', fontweight='bold')

# --- 标注2: 中期优势 (调整了 xytext 和指向弧度) ---
mid_gap = acc_ours[4] - acc_paper[4]
ax.annotate(f'Faster Convergence\n(+{mid_gap:.2f}%)', 
            xy=(5, acc_ours[4]), 
            xycoords='data',
            xytext=(3.0, 92.2), # 移到左上方，不再遮挡曲线趋势
            textcoords='data',
            ha='center',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1", color='#D62728', lw=1.5),
            fontsize=11, color='#D62728', fontweight='bold')

# 自动调整布局，防止边缘裁剪
plt.tight_layout()

# 保存
save_path = 'final_paper_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 最终论文配图已优化生成: {save_path}")
