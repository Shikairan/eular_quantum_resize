#!/usr/bin/env python3
"""
诊断脚本：验证 polarALL_state_int16 与 polarALL_state_int16_cdf 的 vector 实例是否相互污染
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_initial_vec_complex

print("=" * 60)
print("诊断：vector 实例隔离检查")
print("=" * 60)

# 模拟 test_cdfdq 的导入顺序
from polarALL_state_int16 import process_sequence_polar as process_polar_original
import polarALL_state_int16 as orig_module

from polarALL_state_int16_cdf import process_sequence_polar as process_polar_cdf
import polarALL_state_int16_cdf as cdf_module

print(f"\n原始模块 vector id: {id(orig_module.vector)}")
print(f"CDF 模块 vector id:  {id(cdf_module.vector)}")
print(f"两者相同? {orig_module.vector is cdf_module.vector}")

print(f"\n原始 vector 类型: {type(orig_module.vector).__name__} (from {type(orig_module.vector).__module__})")
print(f"CDF vector 类型:   {type(cdf_module.vector).__name__} (from {type(cdf_module.vector).__module__})")

# 检查 process_sequence_polar 引用的 vector（通过 __globals__）
orig_globals_vector = orig_module.process_sequence_polar.__globals__.get('vector')
cdf_globals_vector = cdf_module.process_sequence_polar.__globals__.get('vector')
print(f"\nprocess_polar_original 引用的 vector id: {id(orig_globals_vector) if orig_globals_vector else 'N/A'}")
print(f"process_polar_cdf 引用的 vector id:      {id(cdf_globals_vector) if cdf_globals_vector else 'N/A'}")
print(f"两者引用相同? {orig_globals_vector is cdf_globals_vector}")

# PolarStateEncoded 的 vector_instance
print("\n--- PolarStateEncoded 测试 ---")
initial = create_initial_vec_complex(2**4, False)
state_cdf, _ = process_polar_cdf(initial, [('H', '', [], None, 0)], verbose=False)
state_obj = state_cdf
if hasattr(state_obj, 'vector_instance'):
    print(f"PolarStateEncoded.vector_instance id: {id(state_obj.vector_instance)}")
    print(f"与 cdf_module.vector 相同? {state_obj.vector_instance is cdf_module.vector}")
    print(f"vector_instance 类型: {type(state_obj.vector_instance).__name__}")

print("\n诊断完成。")
