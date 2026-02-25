#!/usr/bin/env python3
"""
运行所有 vector 相关的测试
"""

import subprocess
import sys
import os

def run_test(test_file):
    """运行单个测试文件"""
    print(f"\n{'='*50}")
    print(f"运行测试: {test_file}")
    print('='*50)

    try:
        result = subprocess.run([sys.executable, test_file],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ 测试通过")
            return True
        else:
            print("❌ 测试失败")
            print("错误输出:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试执行出错: {e}")
        return False

def main():
    """主函数"""
    print("运行所有测试")
    print("=" * 60)

    test_files = [
        'test_vector.py',
        'test_vector_precision.py',
        'test_polarALL_state_3.py',
        'test_error_state3.py',
        'test_polarFloat.py',
        'test_polarFloat_simple.py'
    ]

    passed = 0
    total = len(test_files)

    for test_file in test_files:
        if run_test(test_file):
            passed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())