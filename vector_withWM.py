"""
量化欧拉向量类 (PolarVector) - 集成WM变换版本
基于 vector.py 扩展，支持水印变换(WM Transform)功能

主要特性：
- 幅度编码：可选 int16 或 int8，动态范围 [0, 1]，支持缩放
- 相位编码：可选 int16 或 int8，范围 [-π, π]，周期性处理
- 水印变换(WM Transform)：w/m/f变换，自动集成到encode/decode过程
- 兼容 polarALL_state_int16.py 与 README_vector.md 接口
"""

import torch
import math
from typing import List, Tuple, Optional, Union
import sys
import os

# 导入基础vector类用于继承
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vector import PolarVector as BasePolarVector
from cdf.cdf_torch import create_adaptive_quantizer, quantize_adaptive, dequantize_adaptive


class PolarVectorEncoded:
    """
    极坐标编码向量封装类

    存储编码后的幅度向量和相关元数据，支持元组解包
    """

    def __init__(self, r_encoded: torch.Tensor, scale_vec: torch.Tensor,
                 current_max: Optional[float], wm_a: Optional[float], vector_instance: 'PolarVector'):
        """
        初始化PolarVectorEncoded

        Args:
            r_encoded: 编码后的幅度数据 (n_amps,)
            scale_vec: 缩放向量 (n_amps,)
            current_max: 当前最大值（用于WM变换）
            wm_a: WM变换参数a
            vector_instance: 对应的PolarVector实例
        """
        self.r_encoded = r_encoded
        self.scale_vec = scale_vec
        self.current_max = current_max
        self.wm_a = wm_a
        self.vector_instance = vector_instance

    def __iter__(self):
        """支持元组解包：r_encoded, scale_vec = encoded_vector"""
        yield self.r_encoded
        yield self.scale_vec

    def __len__(self):
        """返回向量长度"""
        return len(self.r_encoded)

    def __repr__(self):
        """字符串表示"""
        shape = f"({len(self.r_encoded)},)"
        amplitude_max = f"{self.current_max:.4f}" if self.current_max is not None else "None"
        wm_param = f"{self.wm_a}" if self.wm_a is not None else "None"
        return f"PolarVectorEncoded(shape={shape}, amplitude_max={amplitude_max}, wm_a={wm_param})"

    def get_info(self) -> str:
        """获取信息字符串"""
        info = []
        info.append(f"Shape: ({len(self.r_encoded)},)")
        info.append(f"Scale range: [{self.scale_vec.min().item():.4f}, {self.scale_vec.max().item():.4f}]")
        info.append(f"Current max: {self.current_max:.4f}" if self.current_max is not None else "Current max: None")
        info.append(f"WM parameter a: {self.wm_a}" if self.wm_a is not None else "WM parameter a: None")
        return "\n".join(info)


class PolarStateEncoded:
    """
    完整极坐标状态封装类

    包含极坐标向量和缩放向量，支持完整的状态操作
    """

    def __init__(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                 amplitude_max: float, wm_a: Optional[float], vector_instance: 'PolarVector'):
        """
        初始化PolarStateEncoded

        Args:
            polar_vec: 极坐标向量 (n_amps, 2) - [r_encoded, th_encoded]
            scale_vec: 缩放向量 (n_amps,)
            amplitude_max: 幅度最大值
            wm_a: WM变换参数
            vector_instance: 对应的PolarVector实例
        """
        self.polar_vec = polar_vec
        self.scale_vec = scale_vec
        self.amplitude_max = amplitude_max
        self.wm_a = wm_a
        self.vector_instance = vector_instance

    def __repr__(self):
        """字符串表示"""
        shape = f"({self.polar_vec.shape[0]}, 2)"
        amplitude_max = f"{self.amplitude_max:.4f}"
        wm_param = f"{self.wm_a}" if self.wm_a is not None else "None"
        return f"PolarStateEncoded(shape={shape}, amplitude_max={amplitude_max}, wm_a={wm_param})"

    def get_polar_vec(self) -> torch.Tensor:
        """获取极坐标向量"""
        return self.polar_vec

    def get_scale_vec(self) -> torch.Tensor:
        """获取缩放向量"""
        return self.scale_vec

    def get_amplitude_max(self) -> float:
        """获取幅度最大值"""
        return self.amplitude_max

    def get_wm_a(self) -> Optional[float]:
        """获取WM参数a"""
        return self.wm_a

    def decode_state(self) -> torch.Tensor:
        """
        解码为复数向量，使用存储的 amplitude_max 和 wm_a

        WM解码必须使用每个状态编码时的 amplitude_max，而非 vector 实例的 current_max
        （后者在每次 encode 时会被覆盖，导致历史状态解码错误）

        Returns:
            解码后的复数状态向量
        """
        r_encoded_obj = PolarVectorEncoded(
            self.polar_vec[:, 0], self.scale_vec,
            self.amplitude_max, self.wm_a, self.vector_instance
        )
        r = self.vector_instance.decode_r_tensor(r_encoded_obj).to("cuda:0")
        th = self.vector_instance.decode_th_tensor(self.polar_vec[:, 1])
        #print(r.shape, th.shape, r.device, th.device)
        return torch.complex(r * torch.cos(th), r * torch.sin(th))

    def update_polar_vec(self, new_polar_vec: torch.Tensor, new_scale_vec: torch.Tensor = None,
                         amplitude_max: Optional[float] = None):
        """
        更新极坐标向量

        Args:
            new_polar_vec: 新的极坐标向量
            new_scale_vec: 新的缩放向量（可选，默认为当前值）
            amplitude_max: 可选，新的 amplitude_max（门后更新）
        """
        self.polar_vec = new_polar_vec
        if new_scale_vec is not None:
            self.scale_vec = new_scale_vec
        if amplitude_max is not None:
            self.amplitude_max = amplitude_max

    def clone(self) -> 'PolarStateEncoded':
        """
        克隆PolarStateEncoded对象

        Returns:
            新的PolarStateEncoded对象
        """
        return PolarStateEncoded(
            self.polar_vec.clone(),
            self.scale_vec.clone(),
            self.amplitude_max,
            self.wm_a,
            self.vector_instance
        )

    def get_info(self) -> str:
        """获取信息字符串"""
        info = []
        info.append(f"Shape: {self.polar_vec.shape}")
        info.append(f"Scale range: [{self.scale_vec.min().item():.4f}, {self.scale_vec.max().item():.4f}]")
        info.append(f"Amplitude max: {self.amplitude_max:.4f}")
        info.append(f"WM parameter a: {self.wm_a}" if self.wm_a is not None else "WM parameter a: None")
        return "\n".join(info)

    def __iter__(self):
        """支持元组解包：polar_vec, scale_vec = state"""
        yield self.polar_vec
        yield self.scale_vec


class PolarVector(BasePolarVector):
    """
    集成WM变换的量化欧拉向量类

    在基础PolarVector基础上添加水印变换功能：
    - 自动集成WM变换到encode/decode过程
    - 支持可配置的WM参数a
    - 保持与基础版本的完全兼容性
    """

    def __init__(self, device: Optional[torch.device] = None,
                 amplitude_dtype: torch.dtype = torch.int16,
                 phase_dtype: torch.dtype = torch.int16,
                 wm_a: Optional[float] = 11):
        """
        初始化集成WM变换的极坐标向量类

        Args:
            device: 计算设备，默认为自动检测 (CUDA优先)
            amplitude_dtype: 幅度编码类型，torch.int16 (默认) 或 torch.int8
            phase_dtype: 相位编码类型，torch.int16 (默认) 或 torch.int8
            wm_a: WM变换参数a，默认4.3，设为None禁用WM变换
        """
        # 初始化父类
        super().__init__(device, amplitude_dtype, phase_dtype)

        # WM变换参数
        self.a = wm_a
        self.wm_enabled = wm_a is not None

        # WM变换状态变量
        self.current_max = None

    def get_info(self) -> str:
        """获取配置信息"""
        info_parts = []

        # 基础配置信息
        info_parts.append(f"precision: {str(self.amplitude_dtype).split('.')[-1]} + {str(self.phase_dtype).split('.')[-1]}")
        info_parts.append(f"amplitude_range: '[0.0, 1.0]'")
        info_parts.append(f"phase_range: '[-π, π]'")
        info_parts.append(f"amplitude_dtype: {self.amplitude_dtype}")
        info_parts.append(f"phase_dtype: {self.phase_dtype}")
        info_parts.append(f"device: {self.device}")

        if hasattr(self, 'r_scale'):
            info_parts.append(f"r_scale: {self.r_scale}")
        if hasattr(self, 'TH_SCALE'):
            info_parts.append(f"th_scale: {self.TH_SCALE:.6f}")

        # WM相关信息
        info_parts.append(f"WM_enabled: {self.wm_enabled}")
        if self.wm_enabled:
            info_parts.append(f"WM_parameter_a: {self.a}")
        if self.current_max is not None:
            info_parts.append(f"current_max: {self.current_max:.4f}")

        return ", ".join(info_parts)

    # ===== WM变换核心函数 =====

    def w_max_normalized(self, x: torch.Tensor, a: float, X_max: float) -> torch.Tensor:
        """
        WM变换的w函数：w(x) = e^(a*(x/X_max - 1))

        Args:
            x: 输入幅度值
            a: WM参数
            X_max: 最大值

        Returns:
            WM变换后的值
        """
        self.quantizer = create_adaptive_quantizer(x, bits=16, device=self.device)
        self.q, self.params = quantize_adaptive(x, self.quantizer)
        #print(x.shape, self.q.shape)
        return self.q #torch.log(x/X_max+1) / 0.69315

        #return torch.exp(a * (x - X_max)) #torch.exp(a * (x / X_max - 1))
        #return x
  
    def m_max_normalized(self, x: torch.Tensor, a: float, X_max: float) -> torch.Tensor:
        """
        WM变换的逆变换m函数：m(x) = X_max * (1 + (1/a)*ln(x/scale))

        Args:
            x: 输入值
            a: WM参数
            X_max: 最大值

        Returns:
            逆变换后的值
        """
        scale = 65535.0 if self.amplitude_dtype == torch.int16 else 255.0
        #return x/65535.0   #X_max + (1/a) * torch.log(x / 65535)
        #return X_max + (1/a) * torch.log(x / scale)
        #return X_max*(torch.exp(x*0.69315/scale) - 1)
        return dequantize_adaptive(x, self.params)

    def f_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        WM变换的量化函数f：f(x) = round(scale * x)

        Args:
            x: 输入值

        Returns:
            量化后的整数值
        """
        #scale = 65535.0 if self.amplitude_dtype == torch.int16 else 255.0
        return self.q #torch.round(scale * x).to(self.amplitude_dtype)

    def apply_wm_transform(self, x: torch.Tensor, return_intermediates: bool = False) -> Union[torch.Tensor, dict]:
        """
        手动应用WM变换（用于测试和对比）

        Args:
            x: 输入幅度向量
            return_intermediates: 是否返回中间结果

        Returns:
            变换后的值，或包含中间结果的字典
        """
        if not self.wm_enabled:
            return x

        X_max = torch.max(x).item()

        # 应用WM变换
        w_result = self.w_max_normalized(x, self.a, X_max)
        f_result = self.f_quantize(w_result)
        m_result = self.m_max_normalized(f_result.float(), self.a, X_max)

        if return_intermediates:
            return {
                'X_max': X_max,
                'W': w_result,
                'F': f_result,
                'Y': m_result
            }
        else:

            return m_result

    # ===== 集成WM变换的编解码函数 =====

    def decode_r_tensor(self, r_encoded: Union[torch.Tensor, PolarVectorEncoded],
                        scale_vec: Optional[torch.Tensor] = None,
                        amplitude_max: Optional[float] = None,
                        wm_a: Optional[float] = None) -> torch.Tensor:
        """
        解码幅度 (int16/int8 -> float32，与 polarALL_state_3.py 完全一致)
        如果启用了WM模式，自动应用逆变换还原原始范围

        Args:
            r_encoded: 编码后的幅度数据，或PolarVectorEncoded对象
            scale_vec: 缩放向量（当r_encoded是tensor时需要）
            amplitude_max: 可选，WM解码时的幅度最大值，覆盖 vector.current_max
            wm_a: 可选，WM参数a，覆盖 self.a

        Returns:
            解码后的原始幅度数据
        """
        if isinstance(r_encoded, PolarVectorEncoded):
            # 如果传入的是PolarVectorEncoded对象，直接使用其数据和元信息（显式参数可覆盖）
            r_encoded_tensor = r_encoded.r_encoded
            scale_vec_tensor = r_encoded.scale_vec
            current_max = amplitude_max if amplitude_max is not None else r_encoded.current_max
            wm_a_val = wm_a if wm_a is not None else r_encoded.wm_a
        else:
            # 传统调用方式
            r_encoded_tensor = r_encoded
            scale_vec_tensor = scale_vec
            current_max = amplitude_max if amplitude_max is not None else self.current_max
            wm_a_val = wm_a if wm_a is not None else self.a

        # 解码过程：WM 与 标准 polar 使用不同逻辑
        if current_max is not None and wm_a_val is not None:
            # WM 解码：存储值 q = round(65535*w(r))，直接传入 m 做逆变换
            # int16 溢出：65535 存为 -1，需按无符号解释为 [0, 65535]
            q = r_encoded_tensor.float()
            q = torch.where(q < 0, q + 65536.0, q)
            # 避免 ln(0)：clamp 最小值
            #q = torch.clamp(q, min=1.0)
            r_decoded = self.m_max_normalized(q, wm_a_val, current_max)
        else:
            # 标准 polar 解码过程
            r_base = (r_encoded_tensor.float() + abs(self.r_int_min)) / (self.r_scale - 1.0) * (self.R_MAX - self.R_MIN) + self.R_MIN
            r_decoded = r_base * scale_vec_tensor
        #print("r_decoded shape:",r_decoded.shape)
        return r_decoded

    def encode_r_tensor(self, r: torch.Tensor, amplitude_max: Optional[float] = None) -> PolarVectorEncoded:
        """
        编码幅度 (float32 -> int16/int8 + scale)，与 polarALL_state_3 一致
        若启用了WM模式，自动应用变换。可选传入 amplitude_max 以与同门内其他编码共享（解码时需用同一值）。

        Args:
            r: 幅度向量
            amplitude_max: 可选，WM 编码时的 X_max；不传则用 max(r)
        """
        if self.wm_enabled:
            current_max = amplitude_max if amplitude_max is not None else torch.max(r).item()
            self.current_max = current_max

            # 应用WM变换并量化：w(r) -> f(w(r))
            r_wm = self.w_max_normalized(r, self.a, current_max)
            r_quantized = self.f_quantize(r_wm)
            # WM模式下幅度已归一化，无动态缩放
            scale_vec = torch.ones_like(r, device=self.device)
        else:
            # 不使用WM变换，使用原始polar编码逻辑
            self.current_max = None
            current_max = None

            # 直接使用原始编码逻辑（模拟vector.py的行为）
            scale_vec = torch.ones_like(r, device=self.device)
            large_mask = r > self.R_MAX
            if large_mask.any():
                scale_vec = torch.where(large_mask, r / self.R_MAX, scale_vec)

            r_scaled = r / scale_vec
            r_scaled = torch.clamp(r_scaled, self.R_MIN, self.R_MAX)

            r_scaled_f64 = r_scaled.double()
            encoded_f64 = (r_scaled_f64 - self.R_MIN) / (self.R_MAX - self.R_MIN) * (self.r_scale - 1.0) - abs(self.r_int_min)
            r_quantized = torch.round(encoded_f64).to(self.amplitude_dtype)
            scale_vec = scale_vec

        return PolarVectorEncoded(r_quantized, scale_vec, current_max, self.a if self.wm_enabled else None, self)

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

    def polar_to_complex_tensor(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                               amplitude_max: Optional[float] = None,
                               wm_a: Optional[float] = None) -> torch.Tensor:
        """极坐标 -> 复数；可选传入 amplitude_max、wm_a 用于 WM 解码，覆盖 vector 实例值"""
        #print("polar shape:",polar_vec.shape, scale_vec.shape)
        r = self.decode_r_tensor(polar_vec[:, 0], scale_vec, amplitude_max=amplitude_max, wm_a=wm_a).to("cuda:0")
        th = self.decode_th_tensor(polar_vec[:, 1])

        return torch.complex(r * torch.cos(th), r * torch.sin(th))

    def complex_to_polar_state(self, complex_vec: torch.Tensor,
                               amplitude_max: Optional[float] = None) -> PolarStateEncoded:
        """
        复数 -> 极坐标状态（返回PolarStateEncoded对象）
        amplitude_max: 可选，WM 编码时的共享 X_max，用于门内多路编码一致
        """
        r = torch.abs(complex_vec)
        th = torch.angle(complex_vec)
        r_encoded_obj = self.encode_r_tensor(r, amplitude_max=amplitude_max)
        # 从PolarVectorEncoded对象中提取数据
        r_encoded, scale_vec = r_encoded_obj.r_encoded, r_encoded_obj.scale_vec
        amplitude_max = r_encoded_obj.current_max
        th_encoded = self.encode_th_tensor(th)
        polar_vec = torch.stack([r_encoded, th_encoded], dim=1)

        return PolarStateEncoded(polar_vec, scale_vec, amplitude_max, self.a, self)

    def complex_to_polar_tensor(self, complex_vec: torch.Tensor,
                                amplitude_max: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """复数 -> 极坐标；amplitude_max 可选，用于 WM 门内多路编码共享"""
        state_obj = self.complex_to_polar_state(complex_vec, amplitude_max=amplitude_max)
        return state_obj.polar_vec, state_obj.scale_vec

    # ===== 向量创建与操作 =====

    def create_random_state(self, n_amps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建随机归一化极坐标状态"""
        real_parts = torch.randn(n_amps, device=self.device)
        imag_parts = torch.randn(n_amps, device=self.device)
        complex_vec = torch.complex(real_parts, imag_parts)
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2))
        complex_vec = complex_vec / norm
        return self.complex_to_polar_tensor(complex_vec)

    def to_string(self, polar_vec: torch.Tensor, scale_vec: Optional[torch.Tensor] = None,
                  indices: Optional[List[int]] = None) -> str:
        """转换为字符串表示（用于调试）"""
        if indices is None:
            indices = list(range(len(polar_vec)))

        lines = []
        for i in indices:
            if scale_vec is not None:
                r = self.decode_r_tensor(polar_vec[i:i+1, 0], scale_vec[i:i+1]).item()
            else:
                r = self.decode_r_tensor(polar_vec[i:i+1, 0], torch.ones(1, device=self.device)).item()

            th = self.decode_th_tensor(polar_vec[i:i+1, 1]).item()
            lines.append(f"[{i}] r={r:.6f}, θ={th:.6f} ({th*180/math.pi:.2f}°)")

        return "\n".join(lines)

    def to_complex(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
        """转换为复数表示"""
        return self.polar_to_complex_tensor(polar_vec, scale_vec)

    # ===== 高级功能 =====

    def find_max_amplitude(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                          indices: Optional[List[int]] = None) -> Tuple[int, float]:
        """
        查找向量中的最大幅值元素

        Args:
            polar_vec: 极坐标向量 (n_amps, 2)
            scale_vec: 缩放向量 (n_amps,)
            indices: 要检查的索引列表，默认为全部

        Returns:
            (max_index, max_amplitude): 最大幅值元素的索引和幅值
        """
        if indices is None:
            indices = list(range(len(polar_vec)))

        amplitudes = self.decode_r_tensor(polar_vec[:, 0], scale_vec)
        max_idx = torch.argmax(amplitudes).item()
        max_amp = amplitudes[max_idx].item()

        return max_idx, max_amp


# ===== 模块级函数 =====

def polar_vec_to_string(polar_vec: torch.Tensor, scale_vec: torch.Tensor = None,
                       indices: Optional[List[int]] = None) -> str:
    """转换为字符串表示（模块级函数）"""
    return _get_default_vector().to_string(polar_vec, scale_vec, indices)


def polar_to_complex_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
    """极坐标 -> 复数（模块级函数）"""
    return _get_default_vector().to_complex(polar_vec, scale_vec)


def _get_default_vector() -> PolarVector:
    """获取默认的PolarVector实例"""
    if not hasattr(_get_default_vector, '_instance'):
        _get_default_vector._instance = PolarVector()
    return _get_default_vector._instance


# ===== 向后兼容性函数 =====

def validate_vector(polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                   expected_length: Optional[int] = None) -> bool:
    """
    验证极坐标向量的有效性

    Args:
        polar_vec: 极坐标向量
        scale_vec: 缩放向量
        expected_length: 期望的长度

    Returns:
        是否有效
    """
    if polar_vec.dim() != 2 or polar_vec.shape[1] != 2:
        return False
    if scale_vec.shape != (polar_vec.shape[0],):
        return False
    if expected_length is not None and len(polar_vec) != expected_length:
        return False
    return True
