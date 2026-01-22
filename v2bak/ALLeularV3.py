"""
使用两个 int8 配合欧拉公式描述量子计算中的复数向量 - 混合精度版本

✨ 主要改进：
1. MixedPrecisionState：混合精度表示，对重要状态使用高精度存储
2. HybridScaler：结合一次性温和缩放和渐进式缩放的混合策略
3. 间隔触发：每12个候选门进行一次补偿处理，避免频繁操作

🎯 误差补偿策略：
- 混合精度：15%的最重要状态使用高精度(float32)存储
- 智能缩放：结合渐进式和一次性缩放，避免大幅阶跃
- 间隔处理：减少缩放频率，平衡稳定性和效率

⚠️ 重要修复说明：
1. 原始的 apply_gate_pair 和 apply_polar_gate_pair 函数错误地按相邻元素成对处理状态向量
2. 正确的量子门应该针对指定的比特位进行操作
3. 对于 n 比特系统，状态向量有 2^n 个元素
4. 对第 i 个比特应用单比特门时，需要处理所有满足"第 i 个比特为 0/1"的状态对
5. 这些状态对不是简单的相邻元素，而是根据比特位置有特定的配对规则

例如，对于 2 比特系统 [|00⟩, |01⟩, |10⟩, |11⟩]：
- 对第 0 个比特（最低位）应用 H 门：处理 (|00⟩, |01⟩) 和 (|10⟩, |11⟩)
- 对第 1 个比特应用 H 门：处理 (|00⟩, |10⟩) 和 (|01⟩, |11⟩)

多比特门（CNOT, CZ）也需要根据控制比特和目标比特的状态来正确操作。
"""

import numpy as np
import math
import cmath
import random
from typing import List, Tuple
R_MIN, R_MAX = 0.0, 1.0          # 幅值动态范围 [0,1]
TH_SCALE     = 128 / math.pi     # 相位 [-π,π] → [-128,127]


# ===== 先进的误差补偿类 =====

class MixedPrecisionState:
    """混合精度表示，对重要状态使用更高精度存储"""
    def __init__(self, n_amps, high_precision_ratio=0.15):
        self.high_precision_count = int(n_amps * high_precision_ratio)
        self.low_precision_count = n_amps - self.high_precision_count

        # 高精度存储：最重要的振幅用complex64
        self.high_precision_amps = np.zeros(self.high_precision_count, dtype=np.complex64)
        self.high_precision_indices = set()

        # 低精度存储：其他用int8
        self.low_precision_vec = []

        # 精度切换阈值
        self.importance_threshold = 0.7

    def identify_important_states(self, current_state):
        """识别需要高精度的状态"""
        amplitudes = [(i, abs(amp)) for i, amp in enumerate(current_state)]
        amplitudes.sort(key=lambda x: x[1], reverse=True)

        # 选择幅值最大的状态进行高精度存储
        self.high_precision_indices = set()
        for i, _ in amplitudes[:self.high_precision_count]:
            self.high_precision_indices.add(i)

    def update_high_precision_states(self, current_state):
        """更新高精度状态"""
        for i in self.high_precision_indices:
            idx = list(self.high_precision_indices).index(i)
            self.high_precision_amps[idx] = current_state[i]

    def get_full_state_vector(self):
        """获取完整的状态向量（混合精度）"""
        # 创建低精度版本
        full_vec = self.low_precision_vec.copy()

        # 用高精度数据覆盖重要位置
        for i in self.high_precision_indices:
            idx = list(self.high_precision_indices).index(i)
            full_vec[i] = self.high_precision_amps[idx]

        return full_vec




# ===== 混合缩放策略类 =====

class HybridScaler:
    """结合一次性温和缩放和渐进式缩放的混合策略"""
    def __init__(self, rescale_threshold=1e-3, max_single_scale=5.0, gradual_threshold=0.5, gradual_max_step=1.8):
        self.rescale_threshold = rescale_threshold  # 触发一次性缩放的阈值
        self.max_single_scale = max_single_scale    # 最大一次性缩放倍数
        self.gradual_threshold = gradual_threshold  # 触发渐进式缩放的阈值比例
        self.gradual_max_step = gradual_max_step    # 渐进式缩放的最大步长
        self.target_max = 0.8                       # 目标最大振幅
        self.scale_accumulator = 1.0               # 累积缩放因子

    def should_rescale_single(self, curr_max):
        """判断是否需要一次性缩放"""
        return curr_max < self.rescale_threshold

    def should_rescale_gradual(self, curr_max):
        """判断是否需要渐进式缩放"""
        return curr_max < self.target_max * self.gradual_threshold

    def apply_single_scale(self, byte_vec, curr_max):
        """应用一次性温和缩放"""
        if not self.should_rescale_single(curr_max):
            return False

        # 温和的一次性缩放，避免过大倍数
        scale = min(1.0 / curr_max, self.max_single_scale)

        # 更新累积缩放因子
        self.scale_accumulator *= scale

        # 应用缩放
        for i, (b0, b1) in enumerate(byte_vec):
            r_old = decode_r(b0)
            if r_old == 0:
                continue
            r_new = r_old * scale
            r_new = max(0.0, min(1.0, r_new))
            byte_vec[i] = (encode_r(r_new), b1)

        return True

    def apply_gradual_scale(self, byte_vec, curr_max):
        """应用渐进式缩放"""
        if not self.should_rescale_gradual(curr_max):
            return False

        if curr_max > 0 and curr_max < self.target_max:
            # 渐进式缩放，避免大幅跳跃
            scale_factor = min(self.target_max / curr_max, self.gradual_max_step)
            scale_factor = max(scale_factor, 0.1)

            # 更新累积缩放因子
            self.scale_accumulator *= scale_factor

            # 应用缩放
            for i, (b0, b1) in enumerate(byte_vec):
                r_old = decode_r(b0)
                if r_old > 0:
                    r_new = r_old * scale_factor
                    r_new = min(r_new, 1.0)  # 防止溢出
                    byte_vec[i] = (encode_r(r_new), b1)

        return True

    def apply_hybrid_scale(self, byte_vec):
        """应用混合缩放策略"""
        amps = [decode_r(b0) for b0, _ in byte_vec if decode_r(b0) > 0]
        if not amps:
            return False

        curr_max = max(amps)

        # 优先尝试渐进式缩放
        if self.apply_gradual_scale(byte_vec, curr_max):
            return True

        # 如果渐进式缩放不够，则使用一次性缩放
        if self.apply_single_scale(byte_vec, curr_max):
            return True

        return False


# ===== 编解码函数 =====

def decode_r(b0: int) -> float:
    return (b0 + 128) / 255 * (R_MAX - R_MIN) + R_MIN

def encode_r(r: float) -> int:
    r = max(R_MIN, min(R_MAX, r))
    return int(round((r - R_MIN) / (R_MAX - R_MIN) * 255 - 128))

def decode_th(b1: int) -> float:
    return b1 / TH_SCALE

def encode_th(t: float) -> int:
    t = math.atan2(math.sin(t), math.cos(t))   # wrap [-π,π]
    return int(round(t * TH_SCALE))

def byte2complex(z: Tuple[int, int]) -> complex:
    r, th = decode_r(z[0]), decode_th(z[1])
    return complex(r * math.cos(th), r * math.sin(th))



# ===== 量子门矩阵定义 =====
H_MAT  = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S_MAT  = np.array([[1, 0], [0, 1j]], dtype=complex)
T_MAT  = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
X_MAT  = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MAT  = np.array([[1, 0], [0, -1]], dtype=complex)


# ===== 量子门函数 =====

def hadamard_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return H_MAT @ (z0, z1)

def s_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return S_MAT @ (z0, z1)

def t_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return T_MAT @ (z0, z1)

def x_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return X_MAT @ (z0, z1)

def y_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return Y_MAT @ (z0, z1)

def z_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return Z_MAT @ (z0, z1)

def rx_complex_pair(z0: complex, z1: complex, theta: float) -> Tuple[complex, complex]:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    m = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    return m @ (z0, z1)

def ry_complex_pair(z0: complex, z1: complex, theta: float) -> Tuple[complex, complex]:
    """Ry 门：绕 Y 轴旋转"""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    m = np.array([[c, -s], [s, c]], dtype=complex)
    return m @ (z0, z1)

def rz_complex_pair(z0: complex, z1: complex, phi: float) -> Tuple[complex, complex]:
    """Rz 门：绕 Z 轴旋转"""
    exp_neg = cmath.exp(-1j * phi / 2)
    exp_pos = cmath.exp(1j * phi / 2)
    m = np.array([[exp_neg, 0], [0, exp_pos]], dtype=complex)
    return m @ (z0, z1)

def u1_complex_pair(z0: complex, z1: complex, lambda_param: float) -> Tuple[complex, complex]:
    """U1 门：单参数相位门，等价于 Rz(lambda)"""
    exp_phase = cmath.exp(1j * lambda_param)
    m = np.array([[1, 0], [0, exp_phase]], dtype=complex)
    return m @ (z0, z1)

def u2_complex_pair(z0: complex, z1: complex, phi: float, lambda_param: float) -> Tuple[complex, complex]:
    """U2 门：两参数门"""
    inv_sqrt2 = 1 / math.sqrt(2)
    exp_phi = cmath.exp(1j * phi)
    exp_lambda = cmath.exp(1j * lambda_param)
    m = np.array([[inv_sqrt2, -inv_sqrt2 * exp_lambda],
                  [inv_sqrt2 * exp_phi, inv_sqrt2 * exp_phi * exp_lambda]], dtype=complex)
    return m @ (z0, z1)

def u3_complex_pair(z0: complex, z1: complex, theta: float, phi: float, lambda_param: float) -> Tuple[complex, complex]:
    """U3 门：三参数通用单比特门"""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    exp_phi = cmath.exp(1j * phi)
    exp_lambda = cmath.exp(1j * lambda_param)
    m = np.array([[c, -exp_lambda * s],
                  [exp_phi * s, exp_phi * exp_lambda * c]], dtype=complex)
    return m @ (z0, z1)


# ===== 极坐标量子门 =====

def hadamard_polar_pair(z0: Tuple[int, int], z1: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    a, b = byte2complex(z0), byte2complex(z1)
    inv_sqrt2 = 1 / math.sqrt(2)
    ap = inv_sqrt2 * (a + b)
    bp = inv_sqrt2 * (a - b)
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def x_polar_pair(z0, z1):
    return z1, z0

def y_polar_pair(z0, z1):
    a, b = byte2complex(z0), byte2complex(z1)
    ap = -1j * b
    bp = 1j * a
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def z_polar_pair(z0, z1):
    b0, b1 = z1
    return z0, (b0, encode_th(decode_th(b1) + math.pi))

def s_polar_pair(z0, z1):
    b0, b1 = z1
    return z0, (b0, encode_th(decode_th(b1) + math.pi / 2))

def t_polar_pair(z0, z1):
    b0, b1 = z1
    return z0, (b0, encode_th(decode_th(b1) + math.pi / 4))

def rx_polar_pair(z0, z1, theta: float):
    a, b = byte2complex(z0), byte2complex(z1)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ap = c * a - 1j * s * b
    bp = -1j * s * a + c * b
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def ry_polar_pair(z0, z1, theta: float):
    """Ry 门：绕 Y 轴旋转（极坐标版本）"""
    a, b = byte2complex(z0), byte2complex(z1)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ap = c * a - s * b
    bp = s * a + c * b
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def rz_polar_pair(z0, z1, phi: float):
    """Rz 门：绕 Z 轴旋转（极坐标版本）"""
    a, b = byte2complex(z0), byte2complex(z1)
    exp_neg = cmath.exp(-1j * phi / 2)
    exp_pos = cmath.exp(1j * phi / 2)
    ap = exp_neg * a
    bp = exp_pos * b
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def u1_polar_pair(z0, z1, lambda_param: float):
    """U1 门：单参数相位门（极坐标版本）"""
    a, b = byte2complex(z0), byte2complex(z1)
    exp_phase = cmath.exp(1j * lambda_param)
    ap = a
    bp = exp_phase * b
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def u2_polar_pair(z0, z1, phi: float, lambda_param: float):
    """U2 门：两参数门（极坐标版本）"""
    a, b = byte2complex(z0), byte2complex(z1)
    inv_sqrt2 = 1 / math.sqrt(2)
    exp_phi = cmath.exp(1j * phi)
    exp_lambda = cmath.exp(1j * lambda_param)
    ap = inv_sqrt2 * a - inv_sqrt2 * exp_lambda * b
    bp = inv_sqrt2 * exp_phi * a + inv_sqrt2 * exp_phi * exp_lambda * b
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))

def u3_polar_pair(z0, z1, theta: float, phi: float, lambda_param: float):
    """U3 门：三参数通用单比特门（极坐标版本）"""
    a, b = byte2complex(z0), byte2complex(z1)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    exp_phi = cmath.exp(1j * phi)
    exp_lambda = cmath.exp(1j * lambda_param)
    ap = c * a - exp_lambda * s * b
    bp = exp_phi * s * a + exp_phi * exp_lambda * c * b
    return (encode_r(abs(ap)), encode_th(cmath.phase(ap))), \
           (encode_r(abs(bp)), encode_th(cmath.phase(bp)))


# ===== 控制门处理 =====

def cnot_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return z1, z0

def cz_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return z0, -z1

def cnot_polar_pair(z0, z1):
    return z1, z0

def cz_polar_pair(z0, z1):
    b0, b1 = z1
    return z0, (b0, encode_th(decode_th(b1) + math.pi))


# ===== 门应用函数 =====

def apply_cnot_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int):
    """
    应用 CNOT 门（极坐标版本）

    Args:
        byte_vec: 状态向量
        control_idx: 控制比特索引
        target_idx: 目标比特索引
    """
    n = len(byte_vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    # 对于 CNOT：当控制比特为 |1⟩ 时，翻转目标比特
    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    # 找到所有控制比特为 1 的状态
    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            # 找到对应的目标比特翻转后的状态
            j = i ^ target_mask  # 翻转目标比特
            if i < j:  # 避免重复处理
                byte_vec[i], byte_vec[j] = byte_vec[j], byte_vec[i]

def apply_cz_polar(byte_vec: List[Tuple[int, int]], qubit1_idx: int, qubit2_idx: int):
    """
    应用 CZ 门（极坐标版本）

    Args:
        byte_vec: 状态向量
        qubit1_idx: 第一个比特索引
        qubit2_idx: 第二个比特索引
    """
    n = len(byte_vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert qubit1_idx != qubit2_idx
    assert 0 <= qubit1_idx < nqubit and 0 <= qubit2_idx < nqubit

    # CZ 门：当两个比特都为 |1⟩ 时，相位翻转 π
    mask1 = 1 << qubit1_idx
    mask2 = 1 << qubit2_idx

    for i in range(n):
        if (i & mask1) != 0 and (i & mask2) != 0:  # 两个比特都为 1
            b0, b1 = byte_vec[i]
            byte_vec[i] = (b0, encode_th(decode_th(b1) + math.pi))


# ===== 控制门包装函数 =====

def apply_cy_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CY 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, y_complex_pair)

def apply_cy_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int):
    """应用 CY 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, y_polar_pair)

def apply_ch_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CH 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, hadamard_complex_pair)

def apply_ch_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int):
    """应用 CH 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, hadamard_polar_pair)

def apply_cs_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CS 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, s_complex_pair)

def apply_cs_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int):
    """应用 CS 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, s_polar_pair)

def apply_ct_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CT 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, t_complex_pair)

def apply_ct_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int):
    """应用 CT 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, t_polar_pair)

def apply_crx_complex(vec: List[complex], control_idx: int, target_idx: int, theta: float):
    """应用 CRx 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, rx_complex_pair, theta)

def apply_crx_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, theta: float):
    """应用 CRx 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, rx_polar_pair, theta)

def apply_cry_complex(vec: List[complex], control_idx: int, target_idx: int, theta: float):
    """应用 CRy 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, ry_complex_pair, theta)

def apply_cry_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, theta: float):
    """应用 CRy 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, ry_polar_pair, theta)

def apply_crz_complex(vec: List[complex], control_idx: int, target_idx: int, phi: float):
    """应用 CRz 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, rz_complex_pair, phi)

def apply_crz_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, phi: float):
    """应用 CRz 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, rz_polar_pair, phi)

def apply_cu1_complex(vec: List[complex], control_idx: int, target_idx: int, lambda_param: float):
    """应用 CU1 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, u1_complex_pair, lambda_param)

def apply_cu1_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, lambda_param: float):
    """应用 CU1 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, u1_polar_pair, lambda_param)

def apply_cu2_complex(vec: List[complex], control_idx: int, target_idx: int, phi: float, lambda_param: float):
    """应用 CU2 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, u2_complex_pair, phi, lambda_param)

def apply_cu2_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, phi: float, lambda_param: float):
    """应用 CU2 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, u2_polar_pair, phi, lambda_param)

def apply_cu3_complex(vec: List[complex], control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float):
    """应用 CU3 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, u3_complex_pair, theta, phi, lambda_param)

def apply_cu3_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float):
    """应用 CU3 门（极坐标版本）"""
    apply_controlled_gate_polar(byte_vec, control_idx, target_idx, u3_polar_pair, theta, phi, lambda_param)

def apply_cnot_complex(vec: List[complex], control_idx: int, target_idx: int):
    """
    应用 CNOT 门（复数版本）

    Args:
        vec: 状态向量
        control_idx: 控制比特索引
        target_idx: 目标比特索引
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            j = i ^ target_mask  # 翻转目标比特
            if i < j:  # 避免重复处理
                vec[i], vec[j] = vec[j], vec[i]

def apply_cz_complex(vec: List[complex], qubit1_idx: int, qubit2_idx: int):
    """
    应用 CZ 门（复数版本）

    Args:
        vec: 状态向量
        qubit1_idx: 第一个比特索引
        qubit2_idx: 第二个比特索引
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert qubit1_idx != qubit2_idx
    assert 0 <= qubit1_idx < nqubit and 0 <= qubit2_idx < nqubit

    mask1 = 1 << qubit1_idx
    mask2 = 1 << qubit2_idx

    for i in range(n):
        if (i & mask1) != 0 and (i & mask2) != 0:  # 两个比特都为 1
            vec[i] = -vec[i]

def apply_controlled_gate_complex(vec: List[complex], control_idx: int, target_idx: int, gate_func, *gate_params):
    """
    应用控制门（复数版本）

    Args:
        vec: 状态向量
        control_idx: 控制比特索引
        target_idx: 目标比特索引
        gate_func: 要应用的单比特门函数
        *gate_params: 门函数需要的参数
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    # 找到所有控制比特为 1 的状态，对目标比特应用门
    processed = set()
    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            # 找到目标比特为 0 的状态（清除目标比特位）
            i0 = i & ~target_mask
            # 找到目标比特为 1 的状态（设置目标比特位）
            i1 = i0 | target_mask

            # 确保 i0 和 i1 都满足控制比特为 1
            if (i0 & control_mask) != 0 and (i1 & control_mask) != 0:
                pair_key = (min(i0, i1), max(i0, i1))
                if pair_key not in processed:
                    processed.add(pair_key)
                    z0, z1 = vec[i0], vec[i1]
                    if gate_params:
                        vec[i0], vec[i1] = gate_func(z0, z1, *gate_params)
                    else:
                        vec[i0], vec[i1] = gate_func(z0, z1)

def apply_controlled_gate_polar(byte_vec: List[Tuple[int, int]], control_idx: int, target_idx: int, gate_func, *gate_params):
    """
    应用控制门（极坐标版本）

    Args:
        byte_vec: 状态向量
        control_idx: 控制比特索引
        target_idx: 目标比特索引
        gate_func: 要应用的单比特门函数
        *gate_params: 门函数需要的参数
    """
    n = len(byte_vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    # 找到所有控制比特为 1 的状态，对目标比特应用门
    processed = set()
    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            # 找到目标比特为 0 的状态（清除目标比特位）
            i0 = i & ~target_mask
            # 找到目标比特为 1 的状态（设置目标比特位）
            i1 = i0 | target_mask

            # 确保 i0 和 i1 都满足控制比特为 1
            if (i0 & control_mask) != 0 and (i1 & control_mask) != 0:
                pair_key = (min(i0, i1), max(i0, i1))
                if pair_key not in processed:
                    processed.add(pair_key)
                    z0, z1 = byte_vec[i0], byte_vec[i1]
                    if gate_params:
                        byte_vec[i0], byte_vec[i1] = gate_func(z0, z1, *gate_params)
                    else:
                        byte_vec[i0], byte_vec[i1] = gate_func(z0, z1)


# ===== 核心门应用函数 =====

def apply_polar_gate_pair(byte_vec: List[Tuple[int, int]], gate_func, *args, qubit_idx=0):
    """
    对指定的比特位应用量子门（极坐标版本）

    Args:
        byte_vec: 状态向量，长度为 2^n
        gate_func: 门函数
        *args: 门函数需要的参数（如 theta, phi, lambda 等）
        qubit_idx: 要操作的比特位索引（0 表示最低位）
    """
    n = len(byte_vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n, f"状态向量长度必须是 2 的幂次，当前长度: {n}"
    assert 0 <= qubit_idx < nqubit, f"比特位索引必须在 [0, {nqubit}) 范围内"

    # 计算步长：对于第 qubit_idx 个比特，状态对之间的间隔是 2^qubit_idx
    step = 1 << qubit_idx
    # 每个块的大小是 2^(qubit_idx+1)
    block_size = step << 1

    # 遍历所有需要处理的状态对
    for base in range(0, n, block_size):
        for offset in range(0, step):
            idx0 = base + offset
            idx1 = base + offset + step
            if args:
                byte_vec[idx0], byte_vec[idx1] = gate_func(byte_vec[idx0], byte_vec[idx1], *args)
            else:
                byte_vec[idx0], byte_vec[idx1] = gate_func(byte_vec[idx0], byte_vec[idx1])

def apply_gate_pair(vec: List[complex], gate_func, *args, qubit_idx=0):
    """
    对指定的比特位应用量子门（复数版本）

    Args:
        vec: 状态向量，长度为 2^n
        gate_func: 门函数
        *args: 门函数需要的参数（如 theta, phi, lambda 等）
        qubit_idx: 要操作的比特位索引（0 表示最低位）
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n, f"状态向量长度必须是 2 的幂次，当前长度: {n}"
    assert 0 <= qubit_idx < nqubit, f"比特位索引必须在 [0, {nqubit}) 范围内"

    # 计算步长：对于第 qubit_idx 个比特，状态对之间的间隔是 2^qubit_idx
    step = 1 << qubit_idx
    # 每个块的大小是 2^(qubit_idx+1)
    block_size = step << 1

    # 遍历所有需要处理的状态对
    for base in range(0, n, block_size):
        for offset in range(0, step):
            idx0 = base + offset
            idx1 = base + offset + step
            if args:
                vec[idx0], vec[idx1] = gate_func(vec[idx0], vec[idx1], *args)
            else:
                vec[idx0], vec[idx1] = gate_func(vec[idx0], vec[idx1])


# ===== 先进的缩放和误差处理系统 =====

RESCALE_THRESHOLD = 1e-3  # 可调：论文里用 1e-3~1e-4
RESCALE_CANDIDATE_GATES = ['H', 'Rx', 'Ry', 'U2', 'U3']  # 可能改变幅度的门

# 全局高级误差处理实例
hybrid_scaler = HybridScaler(
    rescale_threshold=RESCALE_THRESHOLD,
    max_single_scale=5.0,      # 最大一次性缩放5倍
    gradual_threshold=0.6,     # 在达到目标的60%时开始渐进缩放
    gradual_max_step=1.5       # 渐进式最大1.5倍
)

mixed_precision_state = MixedPrecisionState(32, high_precision_ratio=0.15)

# 缩放计数器和间隔设置
rescale_counter = 0
RESCALE_INTERVAL = 8  # 每8个候选门后触发一次完整的误差补偿处理

def advanced_error_compensation_processing(name: str, byte_vec: List[Tuple[int, int]], scaler, mixed_precision):
    """
    简化的误差补偿处理：结合混合缩放和混合精度

    Args:
        name: 门名称
        byte_vec: 状态向量
        scaler: HybridScaler实例
        mixed_precision: MixedPrecisionState实例
    """
    global rescale_counter

    if name not in RESCALE_CANDIDATE_GATES:
        return

    rescale_counter += 1

    # 只有在达到间隔时才进行误差补偿处理
    if rescale_counter >= RESCALE_INTERVAL:
        # 1. 记录处理前的状态
        original_states = [byte2complex(z) for z in byte_vec]

        # 2. 更新混合精度状态（识别重要状态）
        mixed_precision.identify_important_states(original_states)
        mixed_precision.low_precision_vec = byte_vec.copy()

        # 3. 应用混合缩放策略
        scaler.apply_hybrid_scale(byte_vec)

        # 4. 计算缩放后的状态，用于更新高精度存储
        scaled_states = [byte2complex(z) for z in byte_vec]
        mixed_precision.update_high_precision_states(scaled_states)

        # 5. 重置计数器
        rescale_counter = 0

        # 调试输出
        # print(f"  [补偿] 处理了 {len(mixed_precision.high_precision_indices)} 个高精度状态")

def get_compensated_state_vector(byte_vec: List[Tuple[int, int]], mixed_precision, residual_comp=None):
    """获取误差补偿后的完整状态向量"""
    n_states = len(byte_vec)

    # 创建补偿后的状态向量
    compensated_vec = [None] * n_states

    # 优先使用高精度状态
    high_precision_count = 0
    for i in mixed_precision.high_precision_indices:
        idx = list(mixed_precision.high_precision_indices).index(i)
        if idx < len(mixed_precision.high_precision_amps):
            compensated_vec[i] = mixed_precision.high_precision_amps[idx]
            high_precision_count += 1

    # 对低精度状态使用基本的解码
    for i, z in enumerate(byte_vec):
        if i not in mixed_precision.high_precision_indices:
            compensated_vec[i] = byte2complex(z)  # 使用基本解码

    # 确保没有None值
    for i in range(n_states):
        if compensated_vec[i] is None:
            compensated_vec[i] = byte2complex(byte_vec[i])

    return compensated_vec


# ===== 门定义和应用 =====

CMP_GATES = {
    'X':  x_complex_pair,
    'Y':  y_complex_pair,
    'S':  s_complex_pair,
    'T':  t_complex_pair,
    'Z':  z_complex_pair,
    'CNOT': cnot_complex_pair,
    'CZ': cz_complex_pair,
    'Rx': rx_complex_pair,
    'Ry': ry_complex_pair,
    'Rz': rz_complex_pair,
    'U1': u1_complex_pair,
    'U2': u2_complex_pair,
    'U3': u3_complex_pair,
    'H': hadamard_complex_pair
}

POLAR_GATES = {
    'Y':  y_polar_pair,
    'X':  x_polar_pair,
    'S':  s_polar_pair,
    'T':  t_polar_pair,
    'Z':  z_polar_pair,
    'CNOT': cnot_polar_pair,
    'CZ': cz_polar_pair,
    'Rx': rx_polar_pair,
    'Ry': ry_polar_pair,
    'Rz': rz_polar_pair,
    'U1': u1_polar_pair,
    'U2': u2_polar_pair,
    'U3': u3_polar_pair,
    'H': hadamard_polar_pair
}


# ===== 序列生成功能 =====

def random_sequence(n: int, m:int, amp_n:int, shuffle:bool=False):
    seq = []
    # 单参数门
    single_param_gates = ['Rx', 'Ry', 'Rz', 'U1']
    # 多参数门
    multi_param_gates = ['U2', 'U3']
    # 无参数单比特门
    no_param_gates = ['X', 'Y', 'S', 'T', 'Z']
    # 无参数控制门
    no_param_controlled_gates = ['CX', 'CY', 'CZ', 'CH', 'CS', 'CT']
    # 单参数控制门
    single_param_controlled_gates = ['CRx', 'CRy', 'CRz', 'CU1']
    # 多参数控制门
    multi_param_controlled_gates = ['CU2', 'CU3']

    # 先生成无参数单比特门
    for _ in range(n):
        g = random.choice(no_param_gates)
        seq.append((g, '', []))

    # 再生成无参数控制门
    for _ in range(n):
        g = random.choice(no_param_controlled_gates)
        seq.append((g, '', []))

    # 再生成有参数单比特门（单参数和多参数）
    for _ in range(m):
        g = random.choice(single_param_gates + multi_param_gates)
        if g in single_param_gates:
            param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({param:.3f})', [param]))
        elif g == 'U2':
            phi = random.uniform(0, 2*math.pi)
            lambda_param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({phi:.3f},{lambda_param:.3f})', [phi, lambda_param]))
        elif g == 'U3':
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, 2*math.pi)
            lambda_param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({theta:.3f},{phi:.3f},{lambda_param:.3f})', [theta, phi, lambda_param]))

    # 再生成有参数控制门
    for _ in range(m):
        g = random.choice(single_param_controlled_gates + multi_param_controlled_gates)
        if g in single_param_controlled_gates:
            param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({param:.3f})', [param]))
        elif g == 'CU2':
            phi = random.uniform(0, 2*math.pi)
            lambda_param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({phi:.3f},{lambda_param:.3f})', [phi, lambda_param]))
        elif g == 'CU3':
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, 2*math.pi)
            lambda_param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({theta:.3f},{phi:.3f},{lambda_param:.3f})', [theta, phi, lambda_param]))

    # 生成幅度相关的门（H 门和旋转门）
    for _ in range(amp_n):
        g = random.choice(['H', 'Rx', 'Ry', 'Rz'])
        if g in ['Rx', 'Ry', 'Rz']:
            param = random.uniform(0, 2*math.pi)
            seq.append((g, f'({param:.3f})', [param]))
        else:
            seq.append((g, '', []))

    if shuffle:
        random.shuffle(seq)
    return seq


# ===== 统一的门应用函数 =====

def apply_gate_unified(vec, byte_vec, gate_name, *gate_params, qubit_idx=0, control_idx=None, target_idx=None):
    """
    统一的量子门应用函数

    Args:
        vec: 复数状态向量
        byte_vec: 极坐标状态向量
        gate_name: 门名称
        *gate_params: 门函数需要的参数
            - Rx, Ry, Rz: 需要一个参数 (theta/phi)
            - U1: 需要一个参数 (lambda)
            - U2: 需要两个参数 (phi, lambda)
            - U3: 需要三个参数 (theta, phi, lambda)
        qubit_idx: 单比特门的比特索引
        control_idx: 控制比特索引（用于控制门）
        target_idx: 目标比特索引（用于控制门）
    """
    # 控制门（需要 control_idx 和 target_idx）
    controlled_gates = {
        'CX': ('CNOT', apply_cnot_complex, apply_cnot_polar),
        'CY': (None, apply_cy_complex, apply_cy_polar),
        'CZ': (None, apply_cz_complex, apply_cz_polar),
        'CH': (None, apply_ch_complex, apply_ch_polar),
        'CS': (None, apply_cs_complex, apply_cs_polar),
        'CT': (None, apply_ct_complex, apply_ct_polar),
        'CRx': (None, apply_crx_complex, apply_crx_polar),
        'CRy': (None, apply_cry_complex, apply_cry_polar),
        'CRz': (None, apply_crz_complex, apply_crz_polar),
        'CU1': (None, apply_cu1_complex, apply_cu1_polar),
        'CU2': (None, apply_cu2_complex, apply_cu2_polar),
        'CU3': (None, apply_cu3_complex, apply_cu3_polar),
    }

    if gate_name in controlled_gates:
        # 控制门
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1

        _, func_complex, func_polar = controlled_gates[gate_name]

        if gate_name == 'CX':
            # CNOT 使用特殊函数
            apply_cnot_complex(vec, control_idx, target_idx)
            apply_cnot_polar(byte_vec, control_idx, target_idx)
        elif gate_name == 'CZ':
            # CZ 使用特殊函数
            apply_cz_complex(vec, control_idx, target_idx)
            apply_cz_polar(byte_vec, control_idx, target_idx)
        else:
            # 其他控制门
            if gate_params:
                func_complex(vec, control_idx, target_idx, *gate_params)
                func_polar(byte_vec, control_idx, target_idx, *gate_params)
            else:
                func_complex(vec, control_idx, target_idx)
                func_polar(byte_vec, control_idx, target_idx)

        # 应用先进的误差补偿
        advanced_error_compensation_processing(gate_name, byte_vec, hybrid_scaler, mixed_precision_state)

    elif gate_name in ['CNOT', 'CZ']:
        # 兼容旧名称
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1
        if gate_name == 'CNOT':
            apply_cnot_complex(vec, control_idx, target_idx)
            apply_cnot_polar(byte_vec, control_idx, target_idx)
        elif gate_name == 'CZ':
            apply_cz_complex(vec, control_idx, target_idx)
            apply_cz_polar(byte_vec, control_idx, target_idx)

        # 应用先进的误差补偿
        advanced_error_compensation_processing(gate_name, byte_vec, hybrid_scaler, mixed_precision_state)

    else:
        # 单比特门
        apply_gate_pair(vec, CMP_GATES[gate_name], *gate_params, qubit_idx=qubit_idx)
        apply_polar_gate_pair(byte_vec, POLAR_GATES[gate_name], *gate_params, qubit_idx=qubit_idx)

        # 应用先进的误差补偿
        advanced_error_compensation_processing(gate_name, byte_vec, hybrid_scaler, mixed_precision_state)


# ===== 测试代码 =====

error_M = []
for kk in range(100):
    N_AMP = 32
    nqubit = int(math.log2(N_AMP))
    cmp_vec = [complex(random.random(), random.random()) for _ in range(N_AMP)]
    nrm = math.sqrt(sum(abs(z)**2 for z in cmp_vec))
    cmp_vec = [z / nrm for z in cmp_vec]

    byte_vec = [(encode_r(abs(z)), encode_th(math.atan2(z.imag, z.real))) for z in cmp_vec]

    seq = random_sequence(150, 150, 150)

    print("Random gate sequence:", [s[0]+s[1] for s in seq])
    print(f"系统有 {nqubit} 个比特，状态向量长度为 {N_AMP}")

    cmp_out = cmp_vec.copy()
    byte_out = byte_vec.copy()
    error = []

    for step,(name, param_str, gate_params) in enumerate(seq):
        # 使用统一的函数应用量子门
        # 随机选择要操作的比特位
        # 检查是否是控制门（以 C 开头，或者是 CNOT/CZ）
        is_controlled = name.startswith('C') or name in ['CNOT', 'CZ']

        if is_controlled:
            # 控制门：随机选择两个不同的比特位
            qubits = random.sample(range(nqubit), 2)
            control_idx, target_idx = qubits[0], qubits[1]
            apply_gate_unified(cmp_out, byte_out, name, *gate_params, qubit_idx=0, control_idx=control_idx, target_idx=target_idx)
            gate_info = f"{name}(控制={control_idx}, 目标={target_idx})"
        else:
            # 单比特门：随机选择一个比特位
            qubit_idx = random.randint(0, nqubit - 1)
            apply_gate_unified(cmp_out, byte_out, name, *gate_params, qubit_idx=qubit_idx, control_idx=None, target_idx=None)
            gate_info = f"{name}(比特={qubit_idx})"

        error_gate = []
        if step == len(seq)-1:
            param_display = param_str if param_str else ''
            print(f"\nAfter {gate_info}{param_display}:")

        # 获取误差补偿后的状态向量
        compensated_vec = get_compensated_state_vector(byte_out, mixed_precision_state)

        for i in range(N_AMP):
            gt_c = cmp_out[i]
            ju_c = compensated_vec[i]  # 使用补偿后的状态向量
            abs_e = abs(gt_c - ju_c)
            rel_e = abs_e / (abs(gt_c) + 1e-15) * 100
            error_gate.append(rel_e)
            if step == len(seq)-1:
                print(f" {i:02b}  {gt_c:>18.5f}  {ju_c:>18.5f}  {abs_e:8.5f} {rel_e:7.2f}%")
        error.append(error_gate)
    error_M.append(error)
error_np = np.array(error_M)
np.save('error_cube_v3.npy', error_np)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- 罇ª轀~B幾T读佅¥ ----
npy_files = list(Path('.').glob('error_cube_v3.npy'))
if len(npy_files) != 1:
    raise FileNotFoundError('弾S佉~M潛®弾U徾E须䷾T住ª罃½彜~I 1 个 .npy 彖~G件')
file_path = npy_files[0]
E = np.load(file_path)          # 形潊¶ (S,M,N)
S, M, N = E.shape
print(f'已佊| 载 {file_path.name}  ->  shape={E.shape}')

# ---- 纾_计 ----
meanMN = E.mean(axis=0)         # (M,N)
stdMN  = E.std(axis=0)          # (M,N)

# 彗¶轗´步维度﻾H对 N 平佝~G﻾I
mean_step = meanMN.mean(axis=1)          # (M,)
std_step  = np.sqrt((stdMN**2).mean(axis=1) / N)

# 潎¯塾C维度﻾H对 M 平佝~G﻾I
mean_env = meanMN.mean(axis=0)           # (N,)
std_env  = np.sqrt((stdMN**2).mean(axis=0) / M)

# ---- 潔»佛¾ ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

x_step = np.arange(M)
ax[0].plot(x_step, mean_step, label='mean')
ax[0].fill_between(x_step, mean_step-std_step, mean_step+std_step,
                   alpha=.25, label='±1 std')
ax[0].set_title('Average error per GATE')
ax[0].set_xlabel('time-step-GATE')
ax[0].set_ylabel('error')
ax[0].grid(True)
ax[0].legend()

x_env = np.arange(N)
ax[1].plot(x_env, mean_env, label='mean')
ax[1].fill_between(x_env, mean_env-std_env, mean_env+std_env,
                   alpha=.25, label='±1 std')
ax[1].set_title('Average error per qubit')
ax[1].set_xlabel('qubit idx')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("3.png")

