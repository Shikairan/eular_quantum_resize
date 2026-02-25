#!/usr/bin/env python3
"""
vector_withWM.py 使用示例

演示如何使用集成了WM变换的PolarVector类
"""

import torch
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_withWM import PolarVector

def main():
    print("=== vector_withWM.py 使用示例 ===\n")

    # 1. 创建WM增强的PolarVector实例
    print("1. 创建PolarVector实例（默认WM参数a=4.3）")
    vector = PolarVector()
    print(f"配置信息: {vector.get_info()}")
    print()

    # 2. 演示WM编码解码
    print("2. WM编码解码演示")
    # 创建测试幅度向量
    amplitudes = torch.tensor([0.1, 0.3, 0.6, 0.8, 0.9], dtype=torch.float32, device=vector.device)
    print(f"原始幅度: {amplitudes}")

    # 编码（自动应用WM变换）
    print("编码过程自动应用WM变换...")
    r_encoded, scale_vec = vector.encode_r_tensor(amplitudes)
    print(f"编码后范围: {r_encoded.min()} 到 {r_encoded.max()}")
    print(f"保存的最大值: {vector.current_max:.4f}")

    # 解码（自动应用WM逆变换）
    print("解码过程自动还原原始范围...")
    r_decoded = vector.decode_r_tensor(r_encoded, scale_vec)
    print(f"解码后幅度: {r_decoded}")

    # 计算重建误差
    error = torch.abs(amplitudes - r_decoded).max()
    print(f"最大重建误差: {error:.8f}")
    print()

    # 3. 比较不同WM参数的效果
    print("3. 比较不同WM参数a的效果")
    a_values = [3.0, 4.3, 5.0]

    for a in a_values:
        vector_a = PolarVector(wm_a=a)
        r_encoded_a, scale_vec_a = vector_a.encode_r_tensor(amplitudes)
        r_decoded_a = vector_a.decode_r_tensor(r_encoded_a, scale_vec_a)
        error_a = torch.abs(amplitudes - r_decoded_a).max()
        print(f"WM参数a={a}: 误差 = {error_a:.8f}")
    print()

    # 4. 完整极坐标向量处理
    print("4. 完整极坐标向量处理示例")
    # 创建复数向量
    complex_vec = torch.complex(amplitudes, torch.randn_like(amplitudes) * 0.2)
    print(f"复数向量实部: {complex_vec.real}")

    # 转换为极坐标（编码过程集成WM）
    polar_vec, scale_vec_full = vector.complex_to_polar_tensor(complex_vec)
    print("转换为极坐标（WM编码）完成")

    # 从极坐标还原（解码过程集成WM逆变换）
    reconstructed_complex = vector.polar_to_complex_tensor(polar_vec, scale_vec_full)
    print(f"还原后实部: {reconstructed_complex.real}")

    # 计算完整重建误差
    full_error = torch.abs(complex_vec - reconstructed_complex).max()
    print(f"完整重建误差: {full_error:.8f}")
    print()

    # 5. 手动WM变换对比
    print("5. 手动WM变换功能仍然可用")
    result = vector.apply_wm_transform(amplitudes, return_intermediates=True)
    print(f"手动WM变换X_max: {result['X_max']:.4f}")
    print(f"手动WM变换误差: {(amplitudes - result['Y']).abs().max():.8f}")
    print()

    print("=== 示例完成 ===")
    print("WM变换已成功集成到PolarVector的编码解码过程中！")

if __name__ == "__main__":
    main()
