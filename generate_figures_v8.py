#!/usr/bin/env python3
"""
为论文 3.3.3 章节重新生成图表（版本 8 - 修复图 3.7 左上角标注问题）
修复问题：
图 3.7：左上角的标注有些问题
可能问题：图例遮挡数据、标注重叠、位置不当等
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置图表样式 - 使用英文字体避免乱码
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.unicode_minus'] = False

# 算法配置
ALGORITHMS = ['MADDPG', 'MAPPO', 'COMA', 'QCOMBO', 'ERNIE', 'SPEAR']
COLORS = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F9D423', '#FF9999', '#FFD93D']

def create_figure_3_7():
    """创建图 3.7：迁移学习性能对比（修复标注问题）"""
    print("重新生成图 3.7：迁移学习性能对比（修复标注问题）...")
    
    # 数据不变
    zero_shot = [45.2, 52.7, 32.7, 38.9, 58.6, 65.3]
    few_shot = [78.3, 85.4, 62.8, 69.5, 88.2, 92.1]
    
    x = np.arange(len(ALGORITHMS))
    width = 0.35
    
    # 创建图表，调整尺寸以适应标注
    plt.figure(figsize=(15, 9))  # 稍微增大图表尺寸
    
    # 绘制柱状图
    bars1 = plt.bar(x - width/2, zero_shot, width, label='Zero-shot Transfer (%)', 
                   color=COLORS, edgecolor='black', linewidth=1.5, alpha=0.9)
    bars2 = plt.bar(x + width/2, few_shot, width, label='Few-shot Fine-tuning (%)', 
                   color=[c + 'CC' for c in COLORS],  # 稍浅的颜色
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # 修复1：调整数值标签位置，避免重叠
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # 根据柱子高度动态调整标签位置
            # 柱子越高，标签位置越高，避免与图例重叠
            if height < 70:
                va_offset = 1  # 较低柱子，标签紧贴顶部
            elif height < 85:
                va_offset = 2  # 中等高度柱子
            else:
                va_offset = 3  # 高柱子，标签更高
            
            plt.text(bar.get_x() + bar.get_width()/2., height + va_offset,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='black')
    
    plt.title('Figure 3.7: Transfer Learning Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Performance (%)', fontsize=14)
    plt.xticks(x, ALGORITHMS, fontsize=12)
    
    # 修复2：调整图例位置和样式
    # 从 'upper left' 改为 'upper right'，避免遮挡左侧数据
    # 降低透明度，增加边框
    plt.legend(loc='upper right', fontsize=12, framealpha=0.85, 
              edgecolor='gray', fancybox=True, shadow=True)
    
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.ylim(0, 105)  # 稍微增加y轴上限，为标注留出空间
    
    # 修复3：调整改进箭头标注
    for i, (z, f) in enumerate(zip(zero_shot, few_shot)):
        improvement = f - z
        
        # 根据改进大小调整箭头位置
        if improvement < 25:
            arrow_y_offset = 6
            text_y_offset = 8
        else:
            arrow_y_offset = 8
            text_y_offset = 10
        
        # 使用更简洁的标注样式
        plt.annotate(f'+{improvement:.1f}%', 
                    xy=(i, f), 
                    xytext=(i, f + arrow_y_offset),
                    ha='center', 
                    fontsize=11, 
                    fontweight='bold', 
                    color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='darkred', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5, connectionstyle="arc3,rad=0.1"))
    
    # 修复4：添加整体说明文本框，放置在左下角
    explanation_text = """Comparison of zero-shot transfer and 
few-shot fine-tuning performance.
Red arrows show improvement from
zero-shot to few-shot."""
    
    plt.text(0.02, 0.02, explanation_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig('figures/图 3.7_迁移学习对比_v8.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 图 3.7 已重新生成（修复标注问题）")
    return 'figures/图 3.7_迁移学习对比_v8.png'

def main():
    print("="*80)
    print("重新生成论文 3.3.3 章节图表（版本 8 - 修复图 3.7 左上角标注问题）")
    print("="*80)
    
    os.makedirs('figures', exist_ok=True)
    
    # 只重新生成图 3.7
    fig_3_7 = create_figure_3_7()
    
    print("\n" + "="*80)
    print("图 3.7 重新生成完成!")
    print("="*80)
    
    print(f"\n生成的图表文件: {fig_3_7}")
    
    # 替换主图表文件
    import shutil
    shutil.copy2(fig_3_7, 'figures/图 3.7_迁移学习对比.png')
    print("已替换 figures/图 3.7_迁移学习对比.png")

if __name__ == "__main__":
    main()