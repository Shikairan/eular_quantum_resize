import math, numpy as np, cmath
from typing import Tuple

# ------------ 刻度工具 ------------
R_MIN, R_MAX = 0.0, 1.0
TH_SCALE = 128 / math.pi

def decode_r(b0: int) -> float:
    return (b0 + 128) / 255 * (R_MAX - R_MIN) + R_MIN

def encode_r(r: float) -> int:
    return int(round(max(R_MIN, min(R_MAX, r)) * 255 - 128))

def decode_th(b1: int) -> float:
    return b1 / TH_SCALE

def encode_th(t: float) -> int:
    return int(round(math.atan2(math.sin(t), math.cos(t)) * TH_SCALE))

def byte2complex(z: Tuple[int, int]) -> complex:
    r, th = decode_r(z[0]), decode_th(z[1])
    return complex(r * math.cos(th), r * math.sin(th))

# ------------ 复数 ground truth ------------
def rx_complex_pair(z0: complex, z1: complex, theta: float):
    c = math.cos(theta / 2); s = math.sin(theta / 2)
    M = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    return M @ np.array([z0, z1])

# ------------ 终极修正版（直接抄） ------------
def rx_polar_pair(z0: Tuple[int, int], z1: Tuple[int, int], theta: float):
    # 1. 字节 → 复数
    c0 = byte2complex(z0)
    c1 = byte2complex(z1)

    # 2. 矩阵乘法
    c, s = math.cos(theta/2), math.sin(theta/2)
    c0p =  c * c0 - 1j * s * c1
    c1p = -1j * s * c0 +  c * c1

    # 3. 复数 → 字节
    return (encode_r(abs(c0p)), encode_th(cmath.phase(c0p))), \
           (encode_r(abs(c1p)), encode_th(cmath.phase(c1p)))
# ------------ 单元测试（固定输入） ------------
if __name__ == "__main__":
    z0 = complex(0.6, 0.2)
    z1 = complex(0.3, -0.1)
    theta = 0.286

    gt0, gt1 = rx_complex_pair(z0, z1, theta)
    byte_z0 = (encode_r(abs(z0)), encode_th(cmath.phase(z0)))
    byte_z1 = (encode_r(abs(z1)), encode_th(cmath.phase(z1)))
    pol0, pol1 = rx_polar_pair(byte_z0, byte_z1, theta)
    pol0_c = byte2complex(pol0)
    pol1_c = byte2complex(pol1)

    print("Unit test Rx(3.052):")
    print(f" ground  z0' = {gt0:+.5f}  z1' = {gt1:+.5f}")
    print(f" polar   z0' = {pol0_c:+.5f}  z1' = {pol1_c:+.5f}")
    print(f" absErr  z0  = {abs(gt0 - pol0_c):.6f}  z1 = {abs(gt1 - pol1_c):.6f}")
