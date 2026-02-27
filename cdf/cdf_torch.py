import torch
import torch.nn.functional as F
from collections import namedtuple
import math
import random
import numpy as np


class AdaptiveHistogramQuantizer:
    """
    基于直方图的自适应分段线性CDF量化器 (PyTorch版本)

    核心思想：
    1. 根据数据分布计算直方图和CDF
    2. 从CDF等间距采样确定自适应断点（输入空间非均匀）
    3. 输出空间均匀分布（最大化信息熵）
    4. 分段线性映射：输入非均匀段 → 输出均匀段
    """

    def __init__(self, n_segments=256, bits=16, hist_bins=10000, device='cpu'):
        """
        Parameters:
            n_segments: 分段数量（通常等于 2^bits - 1）
            bits: 存储位数（8或16），决定输出范围
            hist_bins: 直方图分箱数（细粒度统计）
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.n_segments = n_segments
        self.bits = bits
        self.hist_bins = hist_bins
        self.device = device
        self.max_int = 2**bits - 1

        # 变换参数（由fit方法计算）
        self.t_breaks = None      # 输入断点（自适应，非均匀）
        self.s_breaks = None      # 输出断点（均匀）
        self.slopes = None        # 每段斜率
        self.intercepts = None    # 每段截距
        self.input_range = None   # 输入数据范围

        # CDF查找表（用于快速编码）
        self.cdf_table = None
        self.bin_edges = None

    def fit(self, x):
        """
        学习数据分布，计算自适应分段线性变换

        Parameters:
            x: 训练数据（一维浮点向量），范围[0,1]
        """
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)

        # 记录输入范围
        self.input_range = (x.min().item(), x.max().item())

        # [1] 计算细粒度直方图
        # 范围稍微扩展，避免边界值被截断
        pad = (self.input_range[1] - self.input_range[0]) * 0.001
        hist_range = (max(0, self.input_range[0] - pad),
                      min(1, self.input_range[1] + pad))

        # 使用torch.histc计算直方图
        hist = torch.histc(x, bins=self.hist_bins, min=hist_range[0], max=hist_range[1])
        bin_width = (hist_range[1] - hist_range[0]) / self.hist_bins
        self.bin_edges = torch.linspace(hist_range[0], hist_range[1], self.hist_bins + 1, device=self.device)

        # 转换为密度函数 (相当于np.histogram的density=True)
        # torch.histc返回计数，转换为密度需要除以 (样本数 * bin_width)
        hist = hist / (len(x) * bin_width)

        # [2] 计算CDF（累积分布函数）
        cdf = torch.cumsum(hist, dim=0) * bin_width
        cdf = torch.clamp(cdf, 0, 1)

        # 在CDF开头添加0，确保覆盖完整范围
        cdf = torch.cat([torch.zeros(1, device=self.device), cdf])
        self.cdf_table = cdf

        # [3] 从CDF等间距采样，确定输入断点
        # 目标：在CDF空间均匀分布（即概率质量均匀分布）
        target_cdf_values = torch.linspace(0, 1, self.n_segments + 1, device=self.device)

        # 通过逆CDF找到对应的输入值 (使用numpy确保与原始版本完全一致)
        cdf_np = cdf.cpu().numpy()
        bin_edges_np = self.bin_edges.cpu().numpy()
        target_cdf_np = target_cdf_values.cpu().numpy()
        t_breaks_np = np.interp(target_cdf_np, cdf_np, bin_edges_np)
        self.t_breaks = torch.tensor(t_breaks_np, device=self.device, dtype=torch.float64)

        # 确保边界严格在[0,1]
        self.t_breaks[0] = 0.0
        self.t_breaks[-1] = 1.0

        # [4] 输出断点：均匀分布（最大化利用整数范围）
        # 使用numpy计算确保一致性
        s_breaks_np = np.linspace(0, self.max_int, self.n_segments + 1)
        self.s_breaks = torch.tensor(s_breaks_np, device=self.device, dtype=torch.float64)

        # [5] 预计算每段的线性变换参数 y = a*x + b
        # 使用numpy计算确保与原始版本完全一致
        t_breaks_np = self.t_breaks.cpu().numpy()

        dt_np = t_breaks_np[1:] - t_breaks_np[:-1]
        ds_np = s_breaks_np[1:] - s_breaks_np[:-1]

        # 避免除零
        dt_np = np.where(dt_np < 1e-12, 1e-12, dt_np)

        slopes_np = ds_np / dt_np
        intercepts_np = s_breaks_np[:-1] - slopes_np * t_breaks_np[:-1]

        self.slopes = torch.tensor(slopes_np, device=self.device, dtype=torch.float64)
        self.intercepts = torch.tensor(intercepts_np, device=self.device, dtype=torch.float64)

        return self

    def _interp(self, x, xp, fp):
        """
        PyTorch版本的线性插值函数
        相当于numpy.interp，左边外推使用边界值，右边外推使用边界值
        """
        # 确保输入是tensor
        x = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        xp = torch.as_tensor(xp, device=self.device, dtype=torch.float64)
        fp = torch.as_tensor(fp, device=self.device, dtype=torch.float64)

        # 使用torch.interp (PyTorch 1.11+)
        # torch.interp与np.interp行为一致
        try:
            return torch.interp(x, xp, fp)
        except AttributeError:
            # 如果torch.interp不存在，手动实现
            result = torch.zeros_like(x)
            for i, xi in enumerate(x):
                if xi <= xp[0]:
                    result[i] = fp[0]
                elif xi >= xp[-1]:
                    result[i] = fp[-1]
                else:
                    # 找到xi所在的区间 (左闭右开)
                    idx = torch.searchsorted(xp, xi, right=False) - 1
                    idx = torch.clamp(idx, 0, len(xp) - 2)
                    # 线性插值
                    denom = xp[idx + 1] - xp[idx]
                    if denom == 0:
                        result[i] = fp[idx]
                    else:
                        t = (xi - xp[idx]) / denom
                        result[i] = fp[idx] + t * (fp[idx + 1] - fp[idx])
            return result

    def encode(self, x):
        """
        编码：自适应分段线性变换 Float[0,1] → Int

        使用向量化操作提高效率
        """
        if self.t_breaks is None:
            raise ValueError("必须先调用fit方法学习数据分布")

        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)

        # 裁剪到有效范围
        x_clipped = torch.clamp(x, self.t_breaks[0], self.t_breaks[-1])

        # [1] 找到每个x所属的段索引
        # 使用numpy的searchsorted确保与原始版本完全一致
        t_breaks_np = self.t_breaks.cpu().numpy()
        x_clipped_np = x_clipped.cpu().numpy()
        indices_np = np.searchsorted(t_breaks_np[1:-1], x_clipped_np, side='right')
        indices_np = np.clip(indices_np, 0, self.n_segments - 1)
        indices = torch.tensor(indices_np, device=self.device, dtype=torch.long)

        # [2] 分段线性变换：y = slope * x + intercept
        # 使用numpy计算确保精度一致
        slopes_np = self.slopes.cpu().numpy()
        intercepts_np = self.intercepts.cpu().numpy()
        y_np = slopes_np[indices_np] * x_clipped_np + intercepts_np[indices_np]
        y = torch.tensor(y_np, device=self.device, dtype=torch.float64)

        # [3] 四舍五入到最近整数（与numpy floor(y + 0.5) 等价）
        q_np = np.floor(y_np + 0.5).astype(np.int64)
        q = torch.tensor(q_np, device=self.device, dtype=torch.long)

        # [4] 裁剪到有效整数范围
        q = torch.clamp(q, 0, self.max_int)

        # 转换为存储类型
        if self.bits == 8:
            return q.to(torch.uint8)
        elif self.bits == 16:
            # torch.int16范围不够，需要用int32
            return q.to(torch.int32)
        else:
            return q.to(torch.int32)

    def decode(self, q):
        """
        解码：逆分段线性变换 Int → Float[0,1]

        完全可逆（在量化精度范围内）
        """
        if self.t_breaks is None:
            raise ValueError("必须先调用fit方法")

        q = torch.as_tensor(q, dtype=torch.float64, device=self.device)

        # 裁剪到有效范围
        q_clipped = torch.clamp(q, 0, self.max_int)

        # [1] 找到每个q在输出空间所属的段索引
        # 使用numpy的searchsorted确保一致性
        s_breaks_np = self.s_breaks.cpu().numpy()
        q_clipped_np = q_clipped.cpu().numpy()
        indices_np = np.searchsorted(s_breaks_np[1:-1], q_clipped_np, side='right')
        indices_np = np.clip(indices_np, 0, self.n_segments - 1)
        indices = torch.tensor(indices_np, device=self.device, dtype=torch.long)

        # [2] 逆线性变换：x = (y - intercept) / slope
        x_restored = (q_clipped - self.intercepts[indices]) / self.slopes[indices]

        # [3] 裁剪到[0,1]
        x_restored = torch.clamp(x_restored, 0.0, 1.0)

        return x_restored

    def get_params(self):
        """
        获取变换参数，用于保存和传输
        """
        return {
            'n_segments': self.n_segments,
            'bits': self.bits,
            'max_int': self.max_int,
            't_breaks': self.t_breaks.cpu().numpy() if self.t_breaks is not None else None,      # 关键：自适应输入断点
            's_breaks': self.s_breaks.cpu().numpy() if self.s_breaks is not None else None,      # 均匀输出断点
            'slopes': self.slopes.cpu().numpy() if self.slopes is not None else None,
            'intercepts': self.intercepts.cpu().numpy() if self.intercepts is not None else None,
            'input_range': self.input_range
        }

    def set_params(self, params):
        """
        从保存的参数恢复量化器
        """
        self.n_segments = params['n_segments']
        self.bits = params['bits']
        self.max_int = params['max_int']
        self.t_breaks = torch.tensor(params['t_breaks'], device=self.device) if params['t_breaks'] is not None else None
        self.s_breaks = torch.tensor(params['s_breaks'], device=self.device) if params['s_breaks'] is not None else None
        self.slopes = torch.tensor(params['slopes'], device=self.device) if params['slopes'] is not None else None
        self.intercepts = torch.tensor(params['intercepts'], device=self.device) if params['intercepts'] is not None else None
        self.input_range = params['input_range']
        return self


def create_adaptive_quantizer(data, bits=16, n_segments=None, device='cpu'):
    """
    工厂函数：从数据创建并训练自适应量化器

    Parameters:
        data: 训练数据（代表性样本）
        bits: 8 或 16
        n_segments: 分段数，默认等于 2^bits - 1
        device: 计算设备

    Returns:
        quantizer: 训练好的量化器
    """
    if n_segments is None:
        n_segments = 2**bits - 1

    quantizer = AdaptiveHistogramQuantizer(
        n_segments=n_segments,
        bits=bits,
        device=device
    )
    quantizer.fit(data)

    return quantizer


def quantize_adaptive(x, quantizer_or_data, bits=16, device='cpu'):
    """
    函数式接口：自适应量化

    Parameters:
        x: 待量化数据
        quantizer_or_data: 已训练的量化器，或训练数据
        bits: 如果提供数据，则使用此位数
        device: 计算设备

    Returns:
        q: 量化后的整数
        params: 解码所需的参数
    """
    if isinstance(quantizer_or_data, AdaptiveHistogramQuantizer):
        # 使用已训练的量化器
        quantizer = quantizer_or_data
    else:
        # 从数据训练新量化器
        quantizer = create_adaptive_quantizer(quantizer_or_data, bits=bits, device=device)

    q = quantizer.encode(x)
    params = quantizer.get_params()

    return q, params


def dequantize_adaptive(q, params, device='cpu'):
    """
    函数式接口：自适应反量化

    Parameters:
        q: 量化后的整数
        params: quantize_adaptive返回的参数
        device: 计算设备

    Returns:
        x_restored: 还原的浮点数据
    """
    # 从参数重建量化器
    quantizer = AdaptiveHistogramQuantizer(device=device)
    quantizer.set_params(params)

    return quantizer.decode(q)


def rl_divergence(p, q, eps=1e-12):
    """
    计算KL散度 KL(p||q) = sum(p * log(p/q))

    Parameters:
        p, q: 概率分布（会自动归一化）
        eps: 避免log(0)的小值

    Returns:
        kl_div: KL散度值
    """
    # 确定设备
    device = p.device if isinstance(p, torch.Tensor) else q.device if isinstance(q, torch.Tensor) else 'cpu'

    p = torch.as_tensor(p, dtype=torch.float64, device=device)
    q = torch.as_tensor(q, dtype=torch.float64, device=device)

    # 归一化确保是有效概率分布
    p = p / p.sum()
    q = q / q.sum()

    # 避免零值
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)

    return torch.sum(p * torch.log(p / q)).item()


def run_adaptive_tests(device='cpu'):
    from torch.distributions import Categorical, Normal, kl_divergence
    """
    完整测试自适应直方图量化器 (PyTorch版本)
    包含KL散度对比分析
    """
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("自适应直方图分段线性CDF量化器测试 (PyTorch版本)")
    print("=" * 70)

    # 生成偏斜分布的测试数据（模拟真实概率向量）
    # 混合两个Beta分布：大部分概率集中在0和1附近
    n_samples = 2**25

    # 创建双峰分布（极端偏斜）
    data1 = torch.distributions.Beta(0.3, 5).sample((n_samples // 2,)) * 0.4  # 左偏，集中在0附近
    data2 = 0.6 + torch.distributions.Beta(5, 0.3).sample((n_samples // 2,)) * 0.4  # 右偏，集中在1附近
    O_data = torch.cat([data1, data2], dim=0)
    O_data = torch.clamp(O_data, 0, 1)#.softmax(dim=-1)
    print(O_data.shape, O_data.device, O_data.max(), O_data.min(), torch.sum(O_data))
    # 独立的测试数据（与训练数据相同分布，但使用numpy生成以确保一致性）

    print(f"\n数据分布:")
    print(f"  样本数: {len(O_data)}")
    print(f"  范围: [{O_data.min():.6f}, {O_data.max():.6f}]")
    print(f"  均值: {O_data.mean():.6f}")
    print(f"  标准差: {O_data.std():.6f}")


    results = {}

    # 测试 16-bit 自适应量化
    print("\n" + "-" * 70)
    print("16-bit 自适应直方图量化")
    print("-" * 70)

    quantizer_16 = create_adaptive_quantizer(O_data, bits=16, device=device)
    q16, params16 = quantize_adaptive(O_data, quantizer_16)
    restored_16 = dequantize_adaptive(q16, params16, device=device)
    # 确保测试数据在同一设备上

    # 分析断点分布
    t_breaks = quantizer_16.t_breaks

    print(f"分段数量: {quantizer_16.n_segments}")
    print(f"输入断点范围: [{t_breaks.min():.6f}, {t_breaks.max():.6f}]")

    segment_widths = (t_breaks[1:] - t_breaks[:-1]).cpu().numpy()

    print(f"输入段宽度统计:")
    print(f"  最小: {segment_widths.min():.8f}")
    print(f"  最大: {segment_widths.max():.8f}")
    print(f"  均值: {segment_widths.mean():.8f}")
    print(f"  标准差: {segment_widths.std():.8f}")

    # KL散度分析
    print(f"\nKL散度分析:")

    # 计算原始数据和还原数据的分布（使用直方图近似）
    n_bins_kl = n_samples  # 增加bin数以获得更好的分布近似
    original_hist = torch.histc(O_data, bins=n_bins_kl, min=0, max=1)
    restored_hist = torch.histc(restored_16, bins=n_bins_kl, min=0, max=1)
    kl = torch.sum(O_data.to(device) * (torch.log(O_data.to(device) + 1e-10) - torch.log(restored_16.to(device) + 1e-10)))

    # 使用正确的KL散度计算（基于概率分布）
    kl_adaptive = rl_divergence(original_hist, restored_hist)
    print(f"  自适应量化 KL散度: {kl_adaptive},{kl.item()}")

    # 误差分析
    error_16 = torch.abs(O_data.cpu() - restored_16.cpu())
    print(f"\n量化误差:")
    print(f"\n总误差：{torch.sum(error_16).item():.10f}")
    print(f"  最大误差: {error_16.max():.8f}")
    print(f"  平均误差: {error_16.mean():.10f}")
    print(f"  中位数误差: {torch.median(error_16):.10f}")

    # 与均匀量化对比
    uniform_scale = 65535.0
    q_uniform = torch.round(O_data * uniform_scale).long().to(torch.int16)
    restored_uniform = q_uniform.float() / uniform_scale
    error_uniform = torch.abs(O_data - restored_uniform)

    # 均匀量化的KL散度
    uniform_hist = torch.histc(restored_uniform, bins=n_bins_kl, min=0, max=1)
    kl_uniform = rl_divergence(original_hist, uniform_hist)
    print(f"\n与均匀量化对比:")
    print(f"  平均总误差：{torch.sum(error_uniform).item():.10f}")
    print(f"  自适应总误差：{torch.sum(error_16).item():.10f}")
    print(f"  自适应最大误差: {error_16.max():.8f}")
    print(f"  均匀量化最大误差: {error_uniform.max():.8f}")
    print(f"  自适应平均误差: {error_16.mean():.10f}")
    print(f"  均匀量化平均误差: {error_uniform.mean():.10f}")
    print(f"  自适应 KL散度: {kl_adaptive:.6f}")
    print(f"  均匀量化 KL散度: {kl_uniform:.6f}")
    print(f"  误差改善比例: {(1 - error_16.mean()/error_uniform.mean())*100:.2f}%")
    #print(f"  KL散度改善比例: {(1 - kl_adaptive/kl_uniform)*100:.2f}%")

    print(O_data)
    print(restored_16)
    print(restored_uniform)
    print(q16)
    print(quantizer_16)
    results['int16_adaptive'] = {
        'max_error': error_16.max().item(),
        'mean_error': error_16.mean().item(),
        'kl_divergence': kl_adaptive,
        'segment_widths': segment_widths
    }

    return results, quantizer_16


if __name__ == "__main__":
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results, trained_quantizer = run_adaptive_tests(device=device)