"""
量化欧拉向量类 (PolarVector)
按照 polarALL_state_3.py 的欧拉向量编码要求封装，支持可配置的混合精度

主要特性：
- 幅度编码：可选 int16 或 int8，动态范围 [0, 1]，支持缩放
- 相位编码：可选 int16 或 int8，范围 [-π, π]，周期性处理
- 默认幅值和相位均使用 int16（高精度）
- 完整的 encode/decode 功能封装
- 水印变换(WM Transform)功能：w/m/f变换和幅值搜索
- 兼容 polarALL_state_int16.py 与 README_vector.md 接口
"""

import torch
import math
from typing import List, Tuple, Optional, Union


class PolarVector:
    """
    量化欧拉向量类

    按照 polarALL_state_3.py 的极坐标编码逻辑实现：
    - 幅度 r ∈ [0, 1]，支持动态缩放（r > 1 时用 scale_vec 扩展）
    - 相位 θ ∈ [-π, π]，编码空间内周期性处理
    - 直接在编码空间进行相位运算，避免解码-编码的量化误差（Z/S/T 门优化）
    - 水印变换(WM Transform)：基于最大值归一化的指数/对数变换
    - 幅值搜索：查找向量中的最大幅值元素
    """

    # ===== 类常量 (与 state_3 一致) =====
    R_MIN = 0.0
    R_MAX = 1.0

    def __init__(self, device: Optional[torch.device] = None,
                 amplitude_dtype: torch.dtype = torch.int16,
                 phase_dtype: torch.dtype = torch.int16):
        """
        初始化极坐标向量类

        Args:
            device: 计算设备，默认为自动检测 (CUDA优先)
            amplitude_dtype: 幅度编码类型，torch.int16 (默认) 或 torch.int8
            phase_dtype: 相位编码类型，torch.int16 (默认) 或 torch.int8
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if amplitude_dtype not in [torch.int16, torch.int8]:
            raise ValueError(f"不支持的幅度编码: {amplitude_dtype}，支持: torch.int16, torch.int8")
        if phase_dtype not in [torch.int16, torch.int8]:
            raise ValueError(f"不支持的相位编码: {phase_dtype}，支持: torch.int16, torch.int8")

        self.amplitude_dtype = amplitude_dtype
        self.phase_dtype = phase_dtype

        # 幅度编码参数 (与 state_3 编码逻辑完全一致)
        if amplitude_dtype == torch.int16:
            # 与 polarALL_state_3.py 完全一致：使用 -2048 到 2047 范围
            self.r_int_min = -32768 
            self.r_int_max = 32767
            self.r_scale = 65535.0  # 4096 个量化级别
        else:  # int8
            self.r_int_min = -128
            self.r_int_max = 127
            self.r_scale = 256.0

        # 相位编码参数
        if phase_dtype == torch.int16:
            # 与 polarALL_state_3 一致：[-2048, 2047]
            self.TH_SCALE = 32767 / math.pi
            self.th_int_min = -32768 
            self.th_int_max = 32767
            self.th_period = 65535
        else:  # int8
            self.TH_SCALE = 128 / math.pi
            self.th_int_min = -128
            self.th_int_max = 127
            self.th_period = 256.0

    # ===== 幅度编解码 (遵循 state_3 逻辑) =====

    def decode_r_tensor(self, r_encoded: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
        """解码幅度 (int16/int8 -> float32，与 polarALL_state_3.py 完全一致)"""
        # 与 polarALL_state_3.py 完全一致的解码公式
        r_base = (r_encoded.float() + abs(self.r_int_min)) / (self.r_scale - 1.0) * (self.R_MAX - self.R_MIN) + self.R_MIN
        return r_base * scale_vec

    def encode_r_tensor(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码幅度 (float32 -> int16/int8 + scale)，支持 r > 1 的动态缩放"""
        scale_vec = torch.ones_like(r, device=self.device)
        large_mask = r > self.R_MAX
        if large_mask.any():
            scale_vec = torch.where(large_mask, r / self.R_MAX, scale_vec)

        r_scaled = r / scale_vec
        r_scaled = torch.clamp(r_scaled, self.R_MIN, self.R_MAX)

        r_scaled_f64 = r_scaled.double()
        # 与 polarALL_state_3.py 完全一致的编码公式
        encoded_f64 = (r_scaled_f64 - self.R_MIN) / (self.R_MAX - self.R_MIN) * (self.r_scale - 1.0) - abs(self.r_int_min)
        r_encoded = torch.round(encoded_f64).to(self.amplitude_dtype)

        return r_encoded, scale_vec

    # ===== 相位编解码 (遵循 state_3 的周期性处理) =====

    def decode_th_tensor(self, th_encoded: torch.Tensor) -> torch.Tensor:
        """解码相位 (int8 -> float32)"""
        return th_encoded.float() / self.TH_SCALE

    def encode_th_tensor(self, th: torch.Tensor) -> torch.Tensor:
        """编码相位 (float32 -> int8/int16)"""
        th = torch.atan2(torch.sin(th), torch.cos(th))
        return torch.round(th * self.TH_SCALE).to(self.phase_dtype)

    def add_phase_encoded(self, th_encoded: torch.Tensor, delta_th: float) -> torch.Tensor:
        """
        在编码空间直接添加相位偏移 (与 state_3 add_phase_encoded 逻辑一致)
        避免解码-编码的量化误差，Z/S/T 门的关键优化
        相位 2π 周期性 → 编码空间周期性 (int8: 256, int16: 4096)
        """
        delta_encoded = delta_th * self.TH_SCALE
        new_encoded = th_encoded.float() + delta_encoded
        half_period = self.th_period / 2
        new_encoded_normalized = (new_encoded + half_period) % self.th_period - half_period
        result = torch.round(new_encoded_normalized).to(self.phase_dtype)
        return torch.clamp(result, self.th_int_min, self.th_int_max)

    # ===== 极坐标与复数转换 =====

    def polar_to_complex_tensor(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
        """极坐标 -> 复数"""
        r = self.decode_r_tensor(polar_vec[:, 0], scale_vec)
        th = self.decode_th_tensor(polar_vec[:, 1])
        return torch.complex(r * torch.cos(th), r * torch.sin(th))

    def complex_to_polar_tensor(self, complex_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """复数 -> 极坐标"""
        r = torch.abs(complex_vec)
        th = torch.angle(complex_vec)
        r_encoded, scale_vec = self.encode_r_tensor(r)
        th_encoded = self.encode_th_tensor(th)
        polar_vec = torch.stack([r_encoded, th_encoded], dim=1)
        return polar_vec, scale_vec

    # ===== 向量创建与操作 =====

    def create_random_state(self, n_amps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建随机归一化极坐标状态"""
        real_parts = torch.randn(n_amps, device=self.device)
        imag_parts = torch.randn(n_amps, device=self.device)
        complex_vec = torch.complex(real_parts, imag_parts)
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2))
        complex_vec = complex_vec / norm
        return self.complex_to_polar_tensor(complex_vec)

    def create_from_complex(self, complex_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从复数向量创建（自动归一化）"""
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2))
        complex_vec = complex_vec / norm
        return self.complex_to_polar_tensor(complex_vec)

    def to_string(self, polar_vec: torch.Tensor, scale_vec: Optional[torch.Tensor] = None,
                  indices: Optional[range] = None) -> str:
        """极坐标向量 -> 字符串表示"""
        if indices is None:
            indices = range(len(polar_vec))
        result = []
        for i in indices:
            if scale_vec is not None:
                r = self.decode_r_tensor(polar_vec[i:i+1, 0], scale_vec[i:i+1]).item()
            else:
                r = self.decode_r_tensor(polar_vec[i:i+1, 0], torch.ones(1, device=self.device)).item()
            th = self.decode_th_tensor(polar_vec[i:i+1, 1]).item()
            c = complex(r * math.cos(th), r * math.sin(th))
            result.append(f"{c:.3f}")
        return " | ".join(result)

    def to_complex(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
        """极坐标 -> 复数"""
        return self.polar_to_complex_tensor(polar_vec, scale_vec)

    # ===== 水印变换 (WM Transform) 方法 =====

    def w_max_normalized(self, x: torch.Tensor, a: float = 5.0, X_max: Optional[float] = None) -> torch.Tensor:
        """
        基于最大值归一化的指数变换
        w(x) = exp(a * (x - X_max))

        参数:
            x: 输入张量
            a: 参数，默认值为5.0
            X_max: 预计算的最大值，如果为None则自动计算
        返回:
            变换后的张量
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        if X_max is None:
            X_max = torch.max(x)
        return torch.exp(a * (x - X_max))

    def m_max_normalized(self, x: torch.Tensor, a: float = 5.0, X_max: float = 1.0) -> torch.Tensor:
        """
        基于最大值归一化的对数逆变换
        m(x) = X_max + (1/a) * log(x / 65535)

        参数:
            x: 输入张量
            a: 参数，默认值为5.0
            X_max: 对应的最大值
        返回:
            逆变换后的张量
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        return X_max + (1/a) * torch.log(x / 65535)

    def f_quantize(self, x: torch.Tensor, a: float = 5.0) -> torch.Tensor:
        """
        量化函数
        f(x) = round(65534 * x) + 1

        参数:
            x: 输入张量
            a: 参数（保留兼容性，实际未使用）
        返回:
            量化后的张量
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        return torch.round(65534 * x) + 1

    def find_max_amplitude(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                          top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        搜索极坐标向量中的最大幅值元素

        参数:
            polar_vec: 极坐标向量 (n_amps, 2)
            scale_vec: 缩放向量 (n_amps,)
            top_k: 返回前k个最大值
        返回:
            (max_amplitudes, indices): 最大幅值和对应索引
        """
        # 解码幅度
        amplitudes = self.decode_r_tensor(polar_vec[:, 0], scale_vec)

        # 找到最大值
        if top_k == 1:
            max_amp = torch.max(amplitudes)
            max_idx = torch.argmax(amplitudes)
            return max_amp.unsqueeze(0), max_idx.unsqueeze(0)
        else:
            max_amps, indices = torch.topk(amplitudes, top_k)
            return max_amps, indices

    def find_max_amplitude_complex(self, complex_vec: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从复数向量中搜索最大幅值元素

        参数:
            complex_vec: 复数向量
            top_k: 返回前k个最大值
        返回:
            (max_amplitudes, indices): 最大幅值和对应索引
        """
        amplitudes = torch.abs(complex_vec)

        if top_k == 1:
            max_amp = torch.max(amplitudes)
            max_idx = torch.argmax(amplitudes)
            return max_amp.unsqueeze(0), max_idx.unsqueeze(0)
        else:
            max_amps, indices = torch.topk(amplitudes, top_k)
            return max_amps, indices

    def apply_wm_transform(self, x: torch.Tensor, a: float = 5.0,
                          return_intermediates: bool = False) -> Union[torch.Tensor, dict]:
        """
        应用完整的最大值归一化WM变换

        变换流程:
        X -> w(X) -> f(w(X)) -> m(f(w(X))) = Y

        参数:
            x: 输入向量
            a: 变换参数
            return_intermediates: 是否返回中间结果
        返回:
            如果 return_intermediates=False: 返回变换后的向量 Y
            如果 return_intermediates=True: 返回包含所有中间结果的字典
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        # 计算最大值
        X_max = torch.max(x)

        # 正向变换
        w_X = self.w_max_normalized(x, a, X_max)
        f_w_X = self.f_quantize(w_X, a)

        # 逆向变换
        Y = self.m_max_normalized(f_w_X, a, X_max)

        if return_intermediates:
            return {
                'X': x,
                'X_max': X_max.item(),
                'X_sum': x.sum().item(),
                'w_X': w_X,
                'f_w_X': f_w_X,
                'Y': Y,
                'Y_sum': Y.sum().item(),
                'a': a
            }
        else:
            return Y

    # ===== 工具方法 =====

    def get_info(self) -> dict:
        """获取配置信息"""
        amp_str = str(self.amplitude_dtype).split(".")[-1]
        phase_str = str(self.phase_dtype).split(".")[-1]
        return {
            'precision': f'{amp_str} + {phase_str}',
            'amplitude_range': f'[{self.R_MIN}, {self.R_MAX}]',
            'phase_range': '[-π, π]',
            'amplitude_dtype': str(self.amplitude_dtype),
            'phase_dtype': str(self.phase_dtype),
            'device': str(self.device),
            'th_scale': self.TH_SCALE,
            'r_scale': self.r_scale,
        }

    @staticmethod
    def validate_vector(polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                       amplitude_dtype: torch.dtype = torch.int16,
                       phase_dtype: torch.dtype = torch.int16) -> bool:
        """
        验证极坐标向量有效性
        注：因 torch.stack 可能提升 dtype（如 int8+int16→int16），仅校验数值范围
        """
        if polar_vec.dim() != 2 or polar_vec.shape[1] != 2:
            return False
        if scale_vec.shape != (polar_vec.shape[0],):
            return False
        amp_min, amp_max = (-32768, 32767) if amplitude_dtype == torch.int16 else (-128, 127)
        phase_min, phase_max = (-2048, 2047) if phase_dtype == torch.int16 else (-128, 127)
        if polar_vec[:, 0].min() < amp_min or polar_vec[:, 0].max() > amp_max:
            return False
        if polar_vec[:, 1].min() < phase_min or polar_vec[:, 1].max() > phase_max:
            return False
        return True


# ===== 向后兼容函数 =====

_default_vector: Optional[PolarVector] = None


def _get_default_vector() -> PolarVector:
    global _default_vector
    if _default_vector is None:
        _default_vector = PolarVector()
    return _default_vector


def create_random_polar_state(n_amps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容：创建随机极坐标状态"""
    return _get_default_vector().create_random_state(n_amps)


def polar_vec_to_string(polar_vec: torch.Tensor, scale_vec: torch.Tensor = None,
                       indices=None) -> str:
    """兼容：极坐标转字符串"""
    return _get_default_vector().to_string(polar_vec, scale_vec, indices)


def polar_to_complex_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
    """兼容：极坐标转复数"""
    return _get_default_vector().to_complex(polar_vec, scale_vec)


def complex_to_polar_tensor(complex_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容：复数转极坐标"""
    return _get_default_vector().complex_to_polar_tensor(complex_vec)
