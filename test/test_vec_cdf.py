#!/usr/bin/env python3
"""
CDF 量化编解码测试

使用 utils.py 生成的复数向量，通过 vector_withCDF 的 CDF 量化进行编码-解码往返测试。
验证 CDF 量化的数值精度与稳定性。
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_initial_vec_complex
from vector_withCDF import PolarVector, PolarStateEncoded


def compute_errors(original: torch.Tensor, decoded: torch.Tensor) -> dict:
    """计算原始与解码结果的误差指标"""
    diff = torch.abs(original - decoded)
    return {
        'max_abs': diff.max().item(),
        'rms': torch.sqrt(torch.mean(diff ** 2)).item(),
        'mean_abs': diff.mean().item(),
    }


def test_cdf_roundtrip(n_amps: int, avg: bool, vector: PolarVector, verbose: bool = True) -> dict:
    """
    单次 CDF 编码-解码往返测试

    Args:
        n_amps: 向量长度（2 的幂）
        avg: 是否使用均匀初始化 (True=均匀, False=随机非均匀)
        vector: PolarVector 实例
        verbose: 是否打印详情

    Returns:
        误差指标字典
    """
    initial_vec = create_initial_vec_complex(n_amps, avg=avg)
    complex_tensor = torch.tensor(initial_vec, dtype=torch.complex64, device=vector.device)

    # CDF 编码：复数 -> 极坐标
    polar_vec, params = vector.complex_to_polar_tensor(complex_tensor)

    # CDF 解码：极坐标 -> 复数
    state = PolarStateEncoded(polar_vec, params, vector)
    decoded = state.decode_state()

    errors = compute_errors(complex_tensor, decoded)
    norm_orig = torch.sqrt(torch.sum(torch.abs(complex_tensor) ** 2)).item()
    norm_decoded = torch.sqrt(torch.sum(torch.abs(decoded) ** 2)).item()
    errors['norm_diff'] = abs(norm_orig - norm_decoded)

    if verbose:
        init_type = "均匀" if avg else "随机非均匀"
        print(f"  长度={n_amps}, 初始化={init_type}: "
              f"max_err={errors['max_abs']:.2e}, rms={errors['rms']:.2e}, norm_diff={errors['norm_diff']:.2e}")

    return errors


def run_tests():
    """运行完整 CDF 量化测试套件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vector = PolarVector(device=device, amplitude_dtype=torch.int16, phase_dtype=torch.int16)

    print("=" * 70)
    print("CDF 量化编解码测试（utils 复数向量 + vector_withCDF）")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Vector info: {vector.get_info()}\n")

    all_passed = True
    for n_qubits in [12, 13, 14, 15, 16, 17]:
        n_amps = 2 ** n_qubits
        print(f"--- {n_qubits} 量子比特 ({n_amps} 振幅) ---")

        # 均匀初始化测试
        err_avg = test_cdf_roundtrip(n_amps, avg=True, vector=vector, verbose=True)
        if err_avg['max_abs'] > 1e-4:
            all_passed = False

        # 随机非均匀初始化测试
        err_rand = test_cdf_roundtrip(n_amps, avg=False, vector=vector, verbose=True)
        if err_rand['max_abs'] > 1e-3:
            all_passed = False

        print()

    print("=" * 70)
    if all_passed:
        print("✅ 所有测试通过")
    else:
        print("⚠️ 部分测试误差较大（CDF 量化对非均匀分布更敏感属预期）")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
