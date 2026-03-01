#!/usr/bin/env python3
"""
为论文 3.3.3 章节重新生成图表（版本 9 - 修复图 3.7 颜色对应问题）
修复问题：
图 3.7：标签和直方图的颜色不对应
要求：6个算法的直方图和说明标签对应起来，所有方法只有两个颜色
一个颜色是零样本迁移（深色），另一个颜色是微调迁移（浅色）
要求两个颜色对比明显
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

# 修复：使用两个对比明显的颜色
# 深蓝色 (#1E3A8A) 用于零样本迁移
# 浅蓝色 (#60A5FA) 用于微调迁移
ZERO_SHOT_COLOR = '#1E3A8A'    # 深蓝色
FEW_SHOT_COLOR = '#60A5FA'     # 浅蓝色

def create_figure_3_7():
    """创建图 3.7：迁移学习性能对比（修复颜色对应问题）"""
    print("重新生成图 3.7：迁移学习性能对比（修复颜色对应问题）...")
    
    # 数据不变
    zero_shot = [45.2, 52.7, 32.7, 38.9, 58.6, 65.3]
    few_shot = [78.3, 85.4, 62.8, 69.5, 88.2, 92.1]
    
    x = np.arange(len(ALGORITHMS))
    width = 0.35
    
    # 创建图表，调整尺寸以适应标注
    plt.figure(figsize=(15, 9))
    
    # 修复：所有算法使用相同的颜色对
    # 零样本迁移：深蓝色
    # 微调迁移：浅蓝色
    bars1 = plt.bar(x - width/2, zero_shot, width, 
                   label='Zero-shot Transfer (%)', 
                   color=ZERO_SHOT_COLOR, 
                   edgecolor='black', 
                   linewidth=1.5, 
                   alpha=0.9)
    
    bars2 = plt.bar(x + width/2, few_shot, width, 
                   label='Few-shot Fine-tuning (%)', 
                   color=FEW_SHOT_COLOR, 
                   edgecolor='black', 
                   linewidth=1.5, 
                   alpha=0.9)
    
    # 数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # 根据柱子高度动态调整标签位置
            if height < 70:
                va_offset = 1
            elif height < 85:
                va_offset = 2
            else:
                va_offset = 3
            
            plt.text(bar.get_x() + bar.get_width()/2., height + va_offset,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='black')
    
    plt.title('Figure 3.7: Transfer Learning Performance Comparison', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Performance (%)', fontsize=14)
    plt.xticks(x, ALGORITHMS, fontsize=12)
    
    # 图例
    plt.legend(loc='upper right', fontsize=12, framealpha=0.85, 
              edgecolor='gray', fancybox=True, shadow=True)
    
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.ylim(0, 105)
    
    # 改进箭头标注
    for i, (z, f) in enumerate(zip(zero_shot, few_shot)):
        improvement = f - z
        
        if improvement < 25:
            arrow_y_offset = 6
            text_y_offset = 8
        else:
            arrow_y_offset = 8
            text_y_offset = 10
        
        # 使用与柱子颜色协调的标注颜色
        plt.annotate(f'+{improvement:.1f}%', 
                    xy=(i, f), 
                    xytext=(i, f + arrow_y_offset),
                    ha='center', 
                    fontsize=11, 
                    fontweight='bold', 
                    color='darkblue',
                    bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='lightyellow', 
                            edgecolor='darkblue', 
                            alpha=0.8),
                    arrowprops=dict(arrowstyle='->', 
                                  color='darkblue', 
                                  lw=1.5, 
                                  connectionstyle="arc3,rad=0.1"))
    
    # 添加颜色说明文本框
    explanation_text = """Color Coding:
• Dark Blue (#1E3A8A): Zero-shot Transfer
• Light Blue (#60A5FA): Few-shot Fine-tuning
Red arrows show improvement from
zero-shot to few-shot."""
    
    plt.text(0.02, 0.02, explanation_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    # 添加颜色对比度说明
    contrast_text = """Color Contrast:
• Dark Blue (Zero-shot): Higher contrast for baseline
• Light Blue (Few-shot): Lower contrast for improved performance
Clear visual distinction between two conditions."""
    
    plt.text(0.02, 0.20, contrast_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig('figures/图 3.7_迁移学习对比_v9.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 图 3.7 已重新生成（修复颜色对应问题）")
    
    # 验证颜色使用
    print(f"  零样本迁移颜色: {ZERO_SHOT_COLOR} (深蓝色)")
    print(f"  微调迁移颜色: {FEW_SHOT_COLOR} (浅蓝色)")
    print("  所有 6 个算法使用相同的颜色对")
    
    return 'figures/图 3.7_迁移学习对比_v9.png'

def main():
    print("="*80)
    print("重新生成论文 3.3.3 章节图表（版本 9 - 修复图 3.7 颜色对应问题）")
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