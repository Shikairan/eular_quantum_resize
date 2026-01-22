import numpy as np
import math
import cmath
import random
from typing import List, Tuple

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2.0)
S  = np.array([[1, 0], [0, 1j]])
T  = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]])
Z  = np.array([[1, 0], [0, -1]])
CNOT = np.array([[1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0]], dtype=complex)

CZ   = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, -1]], dtype=complex)

def hadamard_complex_pair(z0: complex, z1: complex):
    return H @ np.array([z0, z1])

def rx_matrix(theta: float):
    c = math.cos(theta / 2); s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])

def hadamard_complex_pair(z0: complex, z1: complex):
    return H @ np.array([z0, z1])

def s_complex_pair(z0: complex, z1: complex):
    return S @ np.array([z0, z1])

def t_complex_pair(z0: complex, z1: complex):
    return T @ np.array([z0, z1])

def z_complex_pair(z0: complex, z1: complex):
    return Z @ np.array([z0, z1])

def rx_complex_pair(z0: complex, z1: complex, theta: float):
    c = math.cos(theta / 2); s = math.sin(theta / 2)
    M = np.array([[c, -1j*s], [-1j*s, c]])
    return M @ np.array([z0, z1])

def cnot_complex_pair(z0: complex, z1: complex):
    # |00〉,|01〉,|10〉,|11〉 里我们只处理 **相邻一对**
    # 这里约定：z0=控制|0〉, z1=目标|1〉 → CNOT 把 |01〉↔|11〉
    # 但“一对”视角下，就是 swap
    return z1, z0

# CZ 只给 |11〉 乘 −1 → 在本对里就是给 z1 乘 −1
def cz_complex_pair(z0: complex, z1: complex):
    return z0, -z1


R_MIN, R_MAX = 0.0, 1.0          # 可随 rescale 更新
TH_SCALE = 128 / math.pi

def decode_r(b0: int) -> float:
    return (b0 + 128) / 255 * (R_MAX - R_MIN) + R_MIN

def encode_r(r: float) -> int:
    clipped = max(R_MIN, min(R_MAX, r))
    return int(round((clipped - R_MIN) / (R_MAX - R_MIN + 1e-30) * 255 - 128))

def decode_th(b1: int) -> float:
    return b1 / TH_SCALE

def encode_th(t: float) -> int:
    return int(round(math.atan2(math.sin(t), math.cos(t)) * TH_SCALE))

def hadamard_gate_pair(z0: Tuple[int, int], z1: Tuple[int, int]):
    c0 = byte2complex(z0)
    c1 = byte2complex(z1)

    inv_sqrt2 = 1/math.sqrt(2)
    c0p = inv_sqrt2 * (c0 + c1)
    c1p = inv_sqrt2 * (c0 - c1)

    # 复数 → 字节
    return (encode_r(abs(c0p)), encode_th(cmath.phase(c0p))), \
           (encode_r(abs(c1p)), encode_th(cmath.phase(c1p)))

# ---------- 相位门 ----------
def s_gate_pair(z0, z1):
    # 只对 target=1 的振幅 +π/2
    b0, b1 = z1
    new_th = decode_th(b1) + math.pi/2
    return z0, (b0, encode_th(new_th))

def t_gate_pair(z0, z1):
    b0, b1 = z1
    new_th = decode_th(b1) + math.pi/4
    return z0, (b0, encode_th(new_th))

def z_gate_pair(z0, z1):
    b0, b1 = z1
    new_th = decode_th(b1) + math.pi
    return z0, (b0, encode_th(new_th))

# ---------- 旋转门 ----------
def rx_polar_pair(z0: Tuple[int, int], z1: Tuple[int, int], theta: float):
    # 1. 嬾W罊~B ὆~R 壾M录°
    c0 = byte2complex(z0)
    c1 = byte2complex(z1)

    # 2. 潟©轘µ举X沾U
    c, s = math.cos(theta/2), math.sin(theta/2)
    c0p =  c * c0 - 1j * s * c1
    c1p = -1j * s * c0 +  c * c1

    # 3. 壾M录° ὆~R 嬾W罊~B
    return (encode_r(abs(c0p)), encode_th(cmath.phase(c0p))), \
           (encode_r(abs(c1p)), encode_th(cmath.phase(c1p)))


# ---------- CNOT ----------
def cnot_pair(z0, z1):
    # 纯置换，字节直接 swap
    return z1, z0

# ---------- CZ ----------
def cz_pair(z0, z1):
    # 只在 target=1 的振幅加 π
    b0, b1 = z1
    new_th = decode_th(b1) + math.pi
    return z0, (b0, encode_th(new_th))



def apply_gate_pair(vec: List[complex], gate_func, theta=None):
    for k in range(0, len(vec), 2):
        if theta is not None:          # 只有 Rx 需要 theta
            vec[k], vec[k+1] = gate_func(vec[k], vec[k+1], theta)
        else:                          # 单比特门只要 2 参数
            vec[k], vec[k+1] = gate_func(vec[k], vec[k+1])


def byte2complex(z: Tuple[int, int]) -> complex:
    r   = decode_r(z[0])      # b0 → 幅度
    th  = decode_th(z[1])     # b1 → 角度
    return complex(r * math.cos(th), r * math.sin(th))

RESCALE_THRESHOLD = 0.2  # 可调：论文里用 1e-3~1e-4
RESCALE_CANDIDATE_GATES = ['H', 'Rx']

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
    'S':  s_complex_pair,
    'T':  t_complex_pair,
    'Z':  z_complex_pair,
    'CNOT': cnot_complex_pair,
    'CZ': cz_complex_pair,
    'Rx': rx_complex_pair,
    'H': hadamard_complex_pair
}

POLAR_GATES = {
    'S':  s_gate_pair,
    'T':  t_gate_pair,
    'Z':  z_gate_pair,
    'CNOT': cnot_pair,
    'CZ': cz_pair,
    'Rx': rx_polar_pair,
    'H': hadamard_gate_pair
}

def random_sequence(n: int):
    seq = []
    for _ in range(n):
        #g = random.choice(['H', 'S','T','Z','CNOT','CZ','Rx'])
        #g = random.choice(['H', 'Rx'])
        g = random.choice(['S','T','Z','CNOT','CZ'])
        if g == 'Rx':
            theta = random.uniform(0, 2*math.pi)
            seq.append(('Rx', f'({theta:.3f})', theta))
        else:
            seq.append((g, '', None))
    return seq

N_AMP = 32
cmp_vec = [complex(random.random(), random.random()) for _ in range(N_AMP)]
nrm = math.sqrt(sum(abs(z)**2 for z in cmp_vec))
cmp_vec = [z / nrm for z in cmp_vec]

byte_vec = [(encode_r(abs(z)), encode_th(math.atan2(z.imag, z.real))) for z in cmp_vec]

seq = random_sequence(1005)
print("Random gate sequence:", [s[0]+s[1] for s in seq])

cmp_out = cmp_vec.copy()
byte_out = byte_vec.copy()


for step,(name, _, theta) in enumerate(seq):
    # 复数路线
    apply_gate_pair(cmp_out, CMP_GATES[name], theta)
    # 极坐标路线
    apply_gate_pair(byte_out, POLAR_GATES[name], theta)
    if step == len(seq)-1:
        print(f"\nAfter {name}{'' if theta is None else f'({theta:.3f})'}:")
    for i in range(N_AMP):
        gt_c = cmp_out[i]
        ju_c = byte2complex(byte_out[i])
        abs_e = abs(gt_c - ju_c)
        rel_e = abs_e / (abs(gt_c) + 1e-15) * 100
        if step == len(seq)-1:
            print(f" {i:02b}  {gt_c:>18.5f}  {ju_c:>18.5f}  {abs_e:8.5f} {rel_e:7.2f}%")
'''
cmp_out = cmp_vec.copy()
byte_out = byte_vec.copy()

for step, (name, _, theta) in enumerate(seq):
    # 复数路线
    apply_gate_pair(cmp_out, CMP_GATES[name], theta)
    # 极坐标路线
    apply_gate_pair(byte_out, POLAR_GATES[name], theta)

    # ✅ 仅候选门后才可能 rescale
    maybe_rescale_after_gate(name, byte_out)

    # ---- 打印当前步结果 & 误差 ----
    if step == len(seq)-1:
        print(f"\nStep {step:02d} | After {name}{'' if theta is None else f'({theta:.3f})'}:")
    for i in range(N_AMP):
        gt_c = cmp_out[i]
        ju_c = byte2complex(byte_out[i])
        abs_e = abs(gt_c - ju_c)
        rel_e = abs_e / (abs(gt_c) + 1e-15) * 100
        if step == len(seq)-1:
            print(f" {i:02b}  {gt_c:>18.5f}  {ju_c:>18.5f}  {abs_e:8.5f} {rel_e:7.2f}%")
'''
