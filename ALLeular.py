"""
使用两个 int8 配合欧拉公式描述量子计算中的复数向量

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

# ----- 编解码 -----
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
H_MAT  = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S_MAT  = np.array([[1, 0], [0, 1j]], dtype=complex)
T_MAT  = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
X_MAT  = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MAT  = np.array([[1, 0], [0, -1]], dtype=complex)

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

# ---------- 极坐标单比特 ----------
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

# 注意：CNOT 和 CZ 是两比特门，不能简单地用 pair 函数处理
# 它们需要根据控制比特的状态来操作目标比特
# 这些函数应该在实际应用时根据控制比特和目标比特的状态来调用

def cnot_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    # ⚠️ 警告：这个函数假设 z0 和 z1 是控制比特为 |1⟩ 时的目标比特状态
    # 正确的 CNOT 应该：当控制比特为 |1⟩ 时，交换目标比特的 |0⟩ 和 |1⟩ 状态
    # 约定：z0=控制为|0⟩时目标的状态，z1=控制为|1⟩时目标的状态
    # 对于 CNOT：|00⟩↔|01⟩ 不变，|10⟩↔|11⟩ 交换
    # 这里假设传入的是需要交换的状态对
    return z1, z0

def cz_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    # ⚠️ 警告：这个函数假设 z0 和 z1 是控制比特为 |1⟩ 时的目标比特状态
    # CZ 门：当两个比特都为 |1⟩ 时，相位翻转
    # 这里假设传入的是 |11⟩ 状态对应的振幅
    return z0, -z1

def cnot_polar_pair(z0, z1):
    # ⚠️ 警告：同 cnot_complex_pair
    return z1, z0

def cz_polar_pair(z0, z1):
    # ⚠️ 警告：同 cz_complex_pair
    b0, b1 = z1
    return z0, (b0, encode_th(decode_th(b1) + math.pi))


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
    # 需要处理所有控制比特为 |1⟩ 的状态对
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


RESCALE_THRESHOLD = 0.2  # 可调：论文里用 1e-3~1e-4
RESCALE_CANDIDATE_GATES = ['H', 'Rx', 'Ry', 'U2', 'U3']  # 可能改变幅度的门

def maybe_rescale_after_gate(name: str, byte_vec: List[Tuple[int, int]]):
    if name not in RESCALE_CANDIDATE_GATES:
        return
    amps = [decode_r(b0) for b0, _ in byte_vec if decode_r(b0) > 0]
    if not amps:
        return
    curr_max = max(amps)
    #print(f"  [DEBUG] {name} 后最大振幅={curr_max:.5f} 阈值={RESCALE_THRESHOLD}")
    gain = 1.0 + (RESCALE_THRESHOLD - curr_max) * 0.2 / RESCALE_THRESHOLD
    gain = min(1.1, gain) 
    if curr_max < RESCALE_THRESHOLD:        # 真·整体衰减
        scale = 1.0 / curr_max               # 只放大，不压扁
        for i, (b0, b1) in enumerate(byte_vec):
            r_old = decode_r(b0)
            if r_old == 0:
                continue
            r_new = r_old * scale#gain            # 线性放大
            r_new = max(0.0, min(1.0, r_new))
            byte_vec[i] = (encode_r(r_new), b1)


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

def random_sequence(n: int, m:int, amp_n:int, shuffle:bool=False):
    seq = []
    # 单参数门
    single_param_gates = ['Rx', 'Ry', 'Rz', 'U1']
    # 多参数门
    multi_param_gates = ['U2', 'U3']
    # 无参数门
    no_param_gates = ['X', 'Y', 'S', 'T', 'Z', 'CNOT', 'CZ']
    
    # 先生成无参数门
    for _ in range(n):
        g = random.choice(no_param_gates)
        seq.append((g, '', []))
    
    # 再生成有参数门（单参数和多参数）
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
        control_idx: CNOT 的控制比特索引
        target_idx: CNOT 的目标比特索引（或 CZ 的第二个比特）
    """
    if gate_name in ['CNOT', 'CZ']:
        # 多比特门
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
    else:
        # 单比特门
        apply_gate_pair(vec, CMP_GATES[gate_name], *gate_params, qubit_idx=qubit_idx)
        apply_polar_gate_pair(byte_vec, POLAR_GATES[gate_name], *gate_params, qubit_idx=qubit_idx)



error_M = []
for kk in range(100):
    N_AMP = 32
    nqubit = int(math.log2(N_AMP))
    cmp_vec = [complex(random.random(), random.random()) for _ in range(N_AMP)]
    nrm = math.sqrt(sum(abs(z)**2 for z in cmp_vec))
    cmp_vec = [z / nrm for z in cmp_vec]

    byte_vec = [(encode_r(abs(z)), encode_th(math.atan2(z.imag, z.real))) for z in cmp_vec]

    seq = random_sequence(150, 2, 2)

    print("Random gate sequence:", [s[0]+s[1] for s in seq])
    print(f"系统有 {nqubit} 个比特，状态向量长度为 {N_AMP}")

    cmp_out = cmp_vec.copy()
    byte_out = byte_vec.copy()
    error = []

    for step,(name, param_str, gate_params) in enumerate(seq):
        # 使用统一的函数应用量子门
        # 随机选择要操作的比特位
        if name in ['CNOT', 'CZ']:
            # 多比特门：随机选择两个不同的比特位
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
        for i in range(N_AMP):
            gt_c = cmp_out[i]
            ju_c = byte2complex(byte_out[i])
            abs_e = abs(gt_c - ju_c)
            rel_e = abs_e / (abs(gt_c) + 1e-15) * 100
            error_gate.append(rel_e)
            if step == len(seq)-1:
                print(f" {i:02b}  {gt_c:>18.5f}  {ju_c:>18.5f}  {abs_e:8.5f} {rel_e:7.2f}%")
        error.append(error_gate)
    error_M.append(error)
error_np = np.aray(error_M)
np.save('error_cube.npy', error_np)  
