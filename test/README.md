# 测试目录

这个目录包含了所有量子计算相关模块的测试文件。

## 目录结构

```
test/
├── test_vector.py                 # PolarVector 基础功能测试
├── test_vector_precision.py       # PolarVector 精度配置测试
├── test_polarALL_state_3.py       # polarALL_state_3 三状态极坐标测试
├── test_error_state3.py           # 三状态版本误差对比测试
├── test_polarFloat.py             # polarFloat 功能测试
├── test_polarFloat_simple.py      # polarFloat 简化测试
├── test_error.py                  # 标准版本误差对比测试
├── test_error_float.py            # float32 版本误差对比测试
├── test_compA_dq.py               # DeepQuantum 兼容性测试
├── test_deepquantum_seq_adapter.py # DeepQuantum 序列适配器测试
├── test_deepquantum_initial_vec.py # DeepQuantum 初始状态测试
├── run_all_tests.py              # 运行所有测试的脚本
└── README.md                     # 本文件
```

## 测试文件说明

### test_vector.py
测试 PolarVector 类的基本功能：
- 基本初始化和配置
- 向量创建和转换
- 精度验证
- 向后兼容性

### test_vector_precision.py
测试不同精度配置的功能：
- int16 高精度模式
- int8 内存优化模式
- 内存使用对比
- 精度和性能分析

## 运行测试

### 运行单个测试
```bash
cd test
python test_vector.py
python test_vector_precision.py
```

### 运行所有测试
```bash
cd test
python run_all_tests.py
```

### 运行特定测试
```bash
cd test
python test_vector.py              # PolarVector 基础功能
python test_vector_precision.py    # PolarVector 精度配置
python test_polarALL_state_3.py    # 三状态极坐标
python test_error_state3.py        # 误差对比测试
```

## 注意事项

- 测试文件使用相对导入路径来导入 vector 模块
- 所有测试都假设 vector.py 在父目录中
- 测试需要 CUDA 支持（如果可用的话）
- 测试会自动检测可用的计算设备

## 添加新测试

1. 在 test/ 目录下创建新的测试文件
2. 在文件开头添加正确的 import 语句：
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   from vector import PolarVector
   ```
3. 在 run_all_tests.py 中添加新的测试文件
4. 更新此 README 文件