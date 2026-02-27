import numpy as np
from collections import namedtuple


class AdaptiveHistogramQuantizer:
    """
    基于直方图的自适应分段线性CDF量化器
    
    核心思想：
    1. 根据数据分布计算直方图和CDF
    2. 从CDF等间距采样确定自适应断点（输入空间非均匀）
    3. 输出空间均匀分布（最大化信息熵）
    4. 分段线性映射：输入非均匀段 → 输出均匀段
    """
    
    def __init__(self, n_segments=256, bits=16, hist_bins=10000):
        """
        Parameters:
            n_segments: 分段数量（通常等于 2^bits - 1）
            bits: 存储位数（8或16），决定输出范围
            hist_bins: 直方图分箱数（细粒度统计）
        """
        self.n_segments = n_segments
        self.bits = bits
        self.hist_bins = hist_bins
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
        x = np.asarray(x, dtype=np.float64)
        
        # 记录输入范围
        self.input_range = (x.min(), x.max())
        
        # [1] 计算细粒度直方图
        # 范围稍微扩展，避免边界值被截断
        pad = (self.input_range[1] - self.input_range[0]) * 0.001
        hist_range = (max(0, self.input_range[0] - pad), 
                      min(1, self.input_range[1] + pad))
        
        hist, self.bin_edges = np.histogram(
            x, 
            bins=self.hist_bins, 
            range=hist_range,
            density=True
        )
        
        # [2] 计算CDF（累积分布函数）
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        cdf = np.cumsum(hist) * bin_width
        cdf = np.clip(cdf, 0, 1)
        
        # 在CDF开头添加0，确保覆盖完整范围
        cdf = np.concatenate([[0], cdf])
        self.cdf_table = cdf
        
        # [3] 从CDF等间距采样，确定输入断点
        # 目标：在CDF空间均匀分布（即概率质量均匀分布）
        target_cdf_values = np.linspace(0, 1, self.n_segments + 1)
        
        # 通过逆CDF找到对应的输入值
        self.t_breaks = np.interp(
            target_cdf_values,    # 目标CDF值
            cdf,                  # 已知CDF值
            self.bin_edges        # 对应的输入值
        )
        
        # 确保边界严格在[0,1]
        self.t_breaks[0] = 0.0
        self.t_breaks[-1] = 1.0
        
        # [4] 输出断点：均匀分布（最大化利用整数范围）
        self.s_breaks = np.linspace(0, self.max_int, self.n_segments + 1)
        
        # [5] 预计算每段的线性变换参数 y = a*x + b
        dt = self.t_breaks[1:] - self.t_breaks[:-1]
        ds = self.s_breaks[1:] - self.s_breaks[:-1]
        
        # 避免除零（理论上dt>0，因为严格单调）
        dt = np.where(dt < 1e-12, 1e-12, dt)
        
        self.slopes = ds / dt
        self.intercepts = self.s_breaks[:-1] - self.slopes * self.t_breaks[:-1]
        
        return self

    def encode(self, x):
        """
        编码：自适应分段线性变换 Float[0,1] → Int
        
        使用向量化操作提高效率
        """
        if self.t_breaks is None:
            raise ValueError("必须先调用fit方法学习数据分布")
        
        x = np.asarray(x, dtype=np.float64)
        
        # 裁剪到有效范围
        x_clipped = np.clip(x, self.t_breaks[0], self.t_breaks[-1])
        
        # [1] 找到每个x所属的段索引
        # searchsorted在有序数组中查找插入位置
        indices = np.searchsorted(self.t_breaks[1:-1], x_clipped, side='right')
        indices = np.clip(indices, 0, self.n_segments - 1)
        
        # [2] 分段线性变换：y = slope * x + intercept
        y = self.slopes[indices] * x_clipped + self.intercepts[indices]
        
        # [3] 四舍五入到最近整数
        q = np.floor(y + 0.5).astype(np.int64)
        
        # [4] 裁剪到有效整数范围
        q = np.clip(q, 0, self.max_int)
        
        # 转换为存储类型
        if self.bits == 8:
            return q.astype(np.uint8)
        else:
            return q.astype(np.uint16)
        
    def decode(self, q):
        """
        解码：逆分段线性变换 Int → Float[0,1]

        完全可逆（在量化精度范围内）
        """
        if self.t_breaks is None:
            raise ValueError("必须先调用fit方法")

        q = np.asarray(q, dtype=np.float64)

        # 裁剪到有效范围
        q_clipped = np.clip(q, 0, self.max_int)

        # [1] 找到每个q在输出空间所属的段索引
        indices = np.searchsorted(self.s_breaks[1:-1], q_clipped, side='right')
        indices = np.clip(indices, 0, self.n_segments - 1)

        # [2] 逆线性变换：x = (y - intercept) / slope
        x_restored = (q_clipped - self.intercepts[indices]) / self.slopes[indices]

        # [3] 裁剪到[0,1]
        x_restored = np.clip(x_restored, 0.0, 1.0)

        return x_restored

    def get_params(self):
        """
        获取变换参数，用于保存和传输
        """
        return {
            'n_segments': self.n_segments,
            'bits': self.bits,
            'max_int': self.max_int,
            't_breaks': self.t_breaks,      # 关键：自适应输入断点
            's_breaks': self.s_breaks,      # 均匀输出断点
            'slopes': self.slopes,
            'intercepts': self.intercepts,
            'input_range': self.input_range
        }

    def set_params(self, params):
        """
        从保存的参数恢复量化器
        """
        self.n_segments = params['n_segments']
        self.bits = params['bits']
        self.max_int = params['max_int']
        self.t_breaks = params['t_breaks']
        self.s_breaks = params['s_breaks']
        self.slopes = params['slopes']
        self.intercepts = params['intercepts']
        self.input_range = params['input_range']
        return self

def create_adaptive_quantizer(data, bits=16, n_segments=None):
    """
    工厂函数：从数据创建并训练自适应量化器

    Parameters:
        data: 训练数据（代表性样本）
        bits: 8 或 16
        n_segments: 分段数，默认等于 2^bits - 1

    Returns:
        quantizer: 训练好的量化器
    """
    if n_segments is None:
        n_segments = 2**bits - 1

    quantizer = AdaptiveHistogramQuantizer(
        n_segments=n_segments,
        bits=bits
    )
    quantizer.fit(data)

    return quantizer


def quantize_adaptive(x, quantizer_or_data, bits=16):
    """
    函数式接口：自适应量化

    Parameters:
        x: 待量化数据
        quantizer_or_data: 已训练的量化器，或训练数据
        bits: 如果提供数据，则使用此位数

    Returns:
        q: 量化后的整数
        params: 解码所需的参数
    """
    if isinstance(quantizer_or_data, AdaptiveHistogramQuantizer):
        # 使用已训练的量化器
        quantizer = quantizer_or_data
    else:
        # 从数据训练新量化器
        quantizer = create_adaptive_quantizer(quantizer_or_data, bits=bits)

    q = quantizer.encode(x)
    params = quantizer.get_params()

    return q, params


def dequantize_adaptive(q, params):
    """
    函数式接口：自适应反量化

    Parameters:
        q: 量化后的整数
        params: quantize_adaptive返回的参数

    Returns:
        x_restored: 还原的浮点数据
    """
    # 从参数重建量化器
    quantizer = AdaptiveHistogramQuantizer()
    quantizer.set_params(params)

    return quantizer.decode(q)


def run_adaptive_tests():
    """
    完整测试自适应直方图量化器
    """
    np.random.seed(42)

    print("=" * 70)
    print("自适应直方图分段线性CDF量化器测试")
    print("=" * 70)

    # 生成偏斜分布的测试数据（模拟真实概率向量）
    # 混合两个Beta分布：大部分概率集中在0和1附近
    n_samples = 2**16
    n_test = 1000

    # 创建双峰分布（极端偏斜）
    data1 = np.random.beta(0.3, 5, n_samples // 2) * 0.4  # 左偏，集中在0附近
    data2 = 0.6 + np.random.beta(5, 0.3, n_samples // 2) * 0.4  # 右偏，集中在1附近
    training_data = np.concatenate([data1, data2])
    training_data = np.clip(training_data, 0, 1)

    # 测试数据
    test_data = np.random.choice(training_data, n_test, replace=False)

    print(f"\n训练数据分布:")
    print(f"  样本数: {len(training_data)}")
    print(f"  范围: [{training_data.min():.6f}, {training_data.max():.6f}]")
    print(f"  均值: {training_data.mean():.6f}")
    print(f"  标准差: {training_data.std():.6f}")

    # 统计各区间分布（验证偏斜）
    bins_check = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist_check, _ = np.histogram(training_data, bins=bins_check)
    print(f"  区间分布: {hist_check / len(training_data) * 100}")

    results = {}

    # 测试 16-bit 自适应量化
    print("\n" + "-" * 70)
    print("16-bit 自适应直方图量化")
    print("-" * 70)

    quantizer_16 = create_adaptive_quantizer(training_data, bits=16)
    q16, params16 = quantize_adaptive(test_data, quantizer_16)
    restored_16 = dequantize_adaptive(q16, params16)

    # 分析断点分布
    t_breaks = quantizer_16.t_breaks
    segment_widths = np.diff(t_breaks)

    print(f"分段数量: {quantizer_16.n_segments}")
    print(f"输入断点范围: [{t_breaks.min():.6f}, {t_breaks.max():.6f}]")
    print(f"输入段宽度统计:")
    print(f"  最小: {segment_widths.min():.8f}")
    print(f"  最大: {segment_widths.max():.8f}")
    print(f"  均值: {segment_widths.mean():.8f}")
    print(f"  标准差: {segment_widths.std():.8f}")

    # 验证：密集区域段更窄（更高分辨率）
    left_dense = np.mean(segment_widths[:50])   # CDF低端（数据密集）
    right_dense = np.mean(segment_widths[-50:]) # CDF高端（数据密集）
    middle_sparse = np.mean(segment_widths[100:-100])  # 中间（数据稀疏）

    print(f"\n自适应效果验证（段宽度越小分辨率越高）:")
    print(f"  左端密集区段宽度: {left_dense:.8f}")
    print(f"  中间稀疏区段宽度: {middle_sparse:.8f}")
    print(f"  右端密集区段宽度: {right_dense:.8f}")

    # 误差分析
    error_16 = np.abs(test_data - restored_16)
    print(f"\n量化误差:")
    print(f"  最大误差: {error_16.max():.8f}")
    print(f"  平均误差: {error_16.mean():.10f}")
    print(f"  中位数误差: {np.median(error_16):.10f}")

    # 与均匀量化对比
    uniform_scale = 65535.0
    q_uniform = np.round(test_data * uniform_scale).astype(np.uint16)
    restored_uniform = q_uniform / uniform_scale
    error_uniform = np.abs(test_data - restored_uniform)

    print(f"\n与均匀量化对比:")
    print(f"  自适应最大误差: {error_16.max():.8f}")
    print(f"  均匀量化最大误差: {error_uniform.max():.8f}")
    print(f"  自适应平均误差: {error_16.mean():.10f}")
    print(f"  均匀量化平均误差: {error_uniform.mean():.10f}")
    print(f"  误差改善比例: {(1 - error_16.mean()/error_uniform.mean())*100:.2f}%")

    results['int16_adaptive'] = {
        'max_error': error_16.max(),
        'mean_error': error_16.mean(),
        'segment_widths': segment_widths
    }

    # 测试 8-bit 自适应量化
    print("\n" + "-" * 70)
    print("8-bit 自适应直方图量化")
    print("-" * 70)

    quantizer_8 = create_adaptive_quantizer(training_data, bits=8)
    q8, params8 = quantize_adaptive(test_data, quantizer_8)
    restored_8 = dequantize_adaptive(q8, params8)

    error_8 = np.abs(test_data - restored_8)
    print(f"分段数量: {quantizer_8.n_segments}")
    print(f"量化误差:")
    print(f"  最大误差: {error_8.max():.6f}")
    print(f"  平均误差: {error_8.mean():.8f}")

    # 8-bit 均匀量化对比
    q_uniform_8 = np.round(test_data * 255).astype(np.uint8)
    restored_uniform_8 = q_uniform_8 / 255.0
    error_uniform_8 = np.abs(test_data - restored_uniform_8)

    print(f"\n与均匀量化对比:")
    print(f"  自适应平均误差: {error_8.mean():.8f}")
    print(f"  均匀量化平均误差: {error_uniform_8.mean():.8f}")

    results['int8_adaptive'] = {
        'max_error': error_8.max(),
        'mean_error': error_8.mean()
    }

    # 可逆性验证
    print("\n" + "-" * 70)
    print("可逆性验证")
    print("-" * 70)

    # 再次编码解码，应该得到相同的整数
    q16_reenc = quantize_adaptive(restored_16, quantizer_16)[0]
    consistency = np.array_equal(q16, q16_reenc)
    print(f"16-bit 再次编码一致性: {consistency}")

    # 参数大小分析
    import sys
    params_size = sys.getsizeof(params16['t_breaks']) + sys.getsizeof(params16['s_breaks'])
    data_size = q16.nbytes
    print(f"\n存储开销:")
    print(f"  量化数据大小: {data_size} bytes")
    print(f"  参数大小（断点表）: {params_size} bytes")
    print(f"  总大小: {data_size + params_size} bytes")
    print(f"  原始float64大小: {test_data.nbytes} bytes")
    print(f"  有效压缩比: {test_data.nbytes / (data_size + params_size):.2f}x")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

    return results, quantizer_16


if __name__ == "__main__":
    results, trained_quantizer = run_adaptive_tests()

