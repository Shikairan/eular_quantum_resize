#!/usr/bin/env python3
"""
测试数值标注修复是否成功的脚本
"""

import sys
import os
import torch
import random
import math

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import analyze_transformation_max_normalized
from KL.wm_simple import create_normalized_vector, analyze_transformation
import matplotlib.pyplot as plt

def test_annotation_fix():
    """测试数值标注修复"""
    print("🔧 Testing Annotation Fix")
    print("=" * 30)

    # 简单的测试数据
    vector_sizes = [16, 32]
    results = {
        'max_normalized': [3.123, 3.456],
        'original': [2.987, 3.234]
    }

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {'max_normalized': 'red', 'original': 'blue'}

    # 绘制简单的数据
    x_positions = [0, 1]
    ax.plot(x_positions, results['max_normalized'], 'o-', color=colors['max_normalized'],
           linewidth=2, markersize=8, label='Max-Normalized')
    ax.plot(x_positions, results['original'], 's-', color=colors['original'],
           linewidth=2, markersize=8, label='Original')

    # 设置x轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['2^4', '2^5'])

    # 添加数值标签 - 这是修复的关键部分
    for i, (max_norm, orig) in enumerate(zip(results['max_normalized'], results['original'])):
        if max_norm is not None:
            ax.annotate(f'{max_norm:.2f}', (i, max_norm),
                       textcoords="offset points", xytext=(0, 10), ha='center',
                       fontsize=12, color=colors['max_normalized'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if orig is not None:
            ax.annotate(f'{orig:.2f}', (i, orig),
                       textcoords="offset points", xytext=(0, -15), ha='center',
                       fontsize=12, color=colors['original'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Vector Size (log₂ scale)')
    ax.set_ylabel('Optimal Parameter a')
    ax.set_title('Annotation Fix Test')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("test_annotation_fix.jpg", dpi=300, bbox_inches='tight')
    print("Test plot saved as: test_annotation_fix.jpg")
    print("Check if the annotations show actual numbers instead of '.2f'")

    plt.close()

if __name__ == "__main__":
    test_annotation_fix()