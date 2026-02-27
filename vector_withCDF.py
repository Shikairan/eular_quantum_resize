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

    def __init__(self, r_encoded: torch.Tensor, params, vector_instance: 'PolarVector'):
        """
        初始化PolarVectorEncoded

        Args:
            r_encoded: 编码后的幅度数据 (n_amps,)
            params: CDF解码参数
            vector_instance: 对应的PolarVector实例
        """
        self.r_encoded = r_encoded
        self.params = params
        self.vector_instance = vector_instance

    def __iter__(self):
        """支持元组解包：r_encoded, scale_vec = encoded_vector"""
        yield self.r_encoded

    def __len__(self):
        """返回向量长度"""
        return len(self.r_encoded)

    def __repr__(self):
        """字符串表示"""
        shape = f"({len(self.r_encoded)},)"
        return f"PolarVectorEncoded(shape={shape})"

    def get_info(self) -> str:
        """获取信息字符串"""
        info = []
        info.append(f"Shape: ({len(self.r_encoded)},)")
        return "\n".join(info)


class PolarStateEncoded:
    """
    完整极坐标状态封装类

    包含极坐标向量和缩放向量，支持完整的状态操作
    """

    def __init__(self, polar_vec: torch.Tensor, params, vector_instance: 'PolarVector'):
        """
        初始化PolarStateEncoded

        Args:
            polar_vec: 极坐标向量 (n_amps, 2) - [r_encoded, th_encoded]
            params: CDF解码参数
            vector_instance: 对应的PolarVector实例
        """
        self.polar_vec = polar_vec
        self.params = params
        self.vector_instance = vector_instance

    def __repr__(self):
        """字符串表示"""
        shape = f"({self.polar_vec.shape[0]}, 2)"
        return f"PolarStateEncoded(shape={shape})"

    def get_polar_vec(self) -> torch.Tensor:
        """获取极坐标向量"""
        return self.polar_vec

    def get_params(self):
        """兼容接口：返回 params（用于 polar_to_complex 解码）"""
        return self.params

    def decode_state(self) -> torch.Tensor:
        """
        解码为复数向量

        Returns:
            解码后的复数状态向量
        """
        r_encoded_obj = PolarVectorEncoded(
            self.polar_vec[:, 0], self.params, self.vector_instance
        )
        r = self.vector_instance.decode_r_tensor(r_encoded_obj).to(self.vector_instance.device)
        th = self.vector_instance.decode_th_tensor(self.polar_vec[:, 1])
        return torch.complex(r * torch.cos(th), r * torch.sin(th))

    def update_polar_vec(self, new_polar_vec: torch.Tensor, new_params=None):
        """
        更新极坐标向量

        Args:
            new_polar_vec: 新的极坐标向量
            new_params: 可选，新的 CDF 参数（若 polar_vec 来自新编码则需更新）
        """
        self.polar_vec = new_polar_vec
        if new_params is not None:
            self.params = new_params

    def clone(self) -> 'PolarStateEncoded':
        """
        克隆PolarStateEncoded对象

        Returns:
            新的PolarStateEncoded对象
        """
        return PolarStateEncoded(
            self.polar_vec.clone(),
            self.params,
            self.vector_instance
        )

    def get_info(self) -> str:
        """获取信息字符串"""
        info = []
        info.append(f"Shape: {self.polar_vec.shape}")
        return "\n".join(info)

    def __iter__(self):
        """支持元组解包：polar_vec, params = state"""
        yield self.polar_vec
        yield self.params


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
                 phase_dtype: torch.dtype = torch.int16):
        """
        初始化集成WM变换的极坐标向量类

        Args:
            device: 计算设备，默认为自动检测 (CUDA优先)
            amplitude_dtype: 幅度编码类型，torch.int16 (默认) 或 torch.int8
            phase_dtype: 相位编码类型，torch.int16 (默认) 或 torch.int8
        """
        # 初始化父类
        super().__init__(device, amplitude_dtype, phase_dtype)
        self.dtype = amplitude_dtype

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

        return ", ".join(info_parts)


    def cdf_float2int(self, x: torch.Tensor) -> torch.Tensor:
        """
        cdf变换的函数

        Args:
            x: 输入幅度值

        Returns:
            WM变换后的值
        """
        self.quantizer = create_adaptive_quantizer(x, bits=16, device=self.device)
        self.q, self.params = quantize_adaptive(x, self.quantizer)
        return self.q 
  
    def cdf_int2float(self, x: torch.Tensor) -> torch.Tensor:
        """
        CDF变换的逆函数（整数 -> 浮点）

        Args:
            x: 量化后的整数值

        Returns:
            反量化后的浮点值
        """
        return dequantize_adaptive(x, self.params, device=str(self.device))

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

    def apply_cdf_transform(self, x: torch.Tensor, return_intermediates: bool = False) -> Union[torch.Tensor, dict]:
        """
        手动应用WM变换（用于测试和对比）

        Args:
            x: 输入幅度向量
            return_intermediates: 是否返回中间结果

        Returns:
            变换后的值，或包含中间结果的字典
        """
        w_result = self.cdf_float2int(x)
        m_result = self.cdf_int2float(w_result.float())

        return m_result

    # ===== 集成WM变换的编解码函数 =====

    def decode_r_tensor(self, r_encoded: Union[torch.Tensor, PolarVectorEncoded],
                        scale_vec: Optional[torch.Tensor] = None,
                        params=None,
                        amplitude_max: Optional[float] = None,
                        wm_a: Optional[float] = None) -> torch.Tensor:
        """
        解码幅度 (CDF量化整数 -> float32)

        Args:
            r_encoded: 编码后的幅度数据，或PolarVectorEncoded对象
            scale_vec: 兼容参数，CDF模式下忽略
            params: CDF解码参数，当r_encoded为裸tensor时必需

        Returns:
            解码后的原始幅度数据
        """
        if isinstance(r_encoded, PolarVectorEncoded):
            r_encoded_tensor = r_encoded.r_encoded
            params = r_encoded.params
        else:
            r_encoded_tensor = r_encoded

        r_decoded = dequantize_adaptive(r_encoded_tensor.float(), params, device=str(self.device))
        return r_decoded.to(device=self.device, dtype=torch.float32)

    def encode_r_tensor(self, r: torch.Tensor, amplitude_max: Optional[float] = None) -> PolarVectorEncoded:
        """幅度编码：float -> CDF量化整数"""
        r_quantized = self.cdf_float2int(r)
        return PolarVectorEncoded(r_quantized, self.params, self)

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

    def polar_to_complex_tensor(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor = None,
                               params=None, amplitude_max: Optional[float] = None,
                               wm_a: Optional[float] = None) -> torch.Tensor:
        """极坐标 -> 复数；CDF 模式需传入 params（来自 complex_to_polar_tensor 的第二返回值）"""
        p = params if params is not None else (scale_vec if isinstance(scale_vec, dict) else None)
        r = self.decode_r_tensor(polar_vec[:, 0], params=p)
        th = self.decode_th_tensor(polar_vec[:, 1]).to(self.device)
        out = torch.complex(r * torch.cos(th), r * torch.sin(th))
        return out.to(device=self.device, dtype=torch.complex64)

    def complex_to_polar_state(self, complex_vec: torch.Tensor,
                               amplitude_max: Optional[float] = None) -> PolarStateEncoded:
        """
        复数 -> 极坐标状态（返回 PolarStateEncoded 对象）
        amplitude_max: 兼容参数，CDF 模式下忽略
        """
        r = torch.abs(complex_vec)
        th = torch.angle(complex_vec)
        r_encoded_obj = self.encode_r_tensor(r, amplitude_max=amplitude_max)
        r_encoded = r_encoded_obj.r_encoded.to(self.device)
        params = r_encoded_obj.params
        th_encoded = self.encode_th_tensor(th).to(self.device)
        polar_vec = torch.stack([r_encoded, th_encoded], dim=1)
        return PolarStateEncoded(polar_vec, params, self)

    def complex_to_polar_tensor(self, complex_vec: torch.Tensor,
                                amplitude_max: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
        """复数 -> 极坐标；返回 (polar_vec, params)，params 用于 polar_to_complex_tensor 解码"""
        state_obj = self.complex_to_polar_state(complex_vec, amplitude_max=amplitude_max)
        return state_obj.polar_vec, state_obj.params

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
                  params=None, indices: Optional[List[int]] = None) -> str:
        """转换为字符串表示（用于调试）；CDF 模式需传入 params"""
        if indices is None:
            indices = list(range(len(polar_vec)))
        p = params if params is not None else (scale_vec if isinstance(scale_vec, dict) else None)
        lines = []
        for i in indices:
            r_enc = polar_vec[i:i+1, 0]
            r_enc_obj = PolarVectorEncoded(r_enc, p, self) if p is not None else None
            r = self.decode_r_tensor(r_enc_obj if r_enc_obj is not None else r_enc, params=p).item()
            th = self.decode_th_tensor(polar_vec[i:i+1, 1]).item()
            lines.append(f"[{i}] r={r:.6f}, θ={th:.6f} ({th*180/math.pi:.2f}°)")
        return "\n".join(lines)

    def to_complex(self, polar_vec: torch.Tensor, scale_vec: torch.Tensor = None,
                   params=None) -> torch.Tensor:
        """转换为复数表示；CDF 模式传入 params（来自 complex_to_polar_tensor 的第二返回值）"""
        return self.polar_to_complex_tensor(polar_vec, scale_vec=scale_vec, params=params)


# ===== 模块级函数 =====

def polar_vec_to_string(polar_vec: torch.Tensor, scale_vec: torch.Tensor = None,
                       params=None, indices: Optional[List[int]] = None) -> str:
    """转换为字符串表示（模块级函数）；CDF 模式传入 params"""
    return _get_default_vector().to_string(polar_vec, scale_vec, params, indices)


def polar_to_complex_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor = None,
                            params=None) -> torch.Tensor:
    """极坐标 -> 复数（模块级函数）；CDF 模式传入 params"""
    return _get_default_vector().to_complex(polar_vec, scale_vec, params)


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
