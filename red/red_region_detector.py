"""
红色区域检测模块

使用OpenCV检测图片中的红色区域，支持参数调整。

使用示例:
    import cv2
    import numpy as np
    from red_region_detector import detect_red_regions, draw_red_regions

    # 读取图片
    image = cv2.imread('image.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 检测红色区域
    red_regions = detect_red_regions(
        image_rgb,
        red_lower=(0, 30, 30),        # 红色下限阈值
        red_upper=(10, 255, 255),     # 红色上限阈值
        area_threshold_ratio=0.05,     # 5%面积阈值
        min_contour_area=100          # 最小轮廓面积
    )

    # 输出结果
    print(f"检测到 {len(red_regions)} 个红色区域")

    # 可视化结果
    result_image = draw_red_regions(image_rgb, red_regions)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def detect_red_regions(
    image: np.ndarray,
    red_lower: Tuple[int, int, int] = (0, 150, 150),
    red_upper: Tuple[int, int, int] = (10, 255, 255),
    area_threshold_ratio: float = 0.05,
    min_contour_area: int = 100
) -> List[List[Tuple[int, int]]]:
    """
    检测图片中的红色区域，返回连成片的红色区域坐标列表

    检测策略：
    1. 只在图片下方66%的区域进行检测（上方34%完全忽略）
    2. 红色区域需要连成片
    3. 区域面积需要大于指定的阈值比例（相对于下方66%区域的面积）

    Args:
        image: RGB格式的numpy数组图片
        red_lower: HSV颜色空间中红色下限阈值 (H, S, V)
        red_upper: HSV颜色空间中红色上限阈值 (H, S, V)
        area_threshold_ratio: 面积阈值比例，相对于下方66%区域的面积 (默认5%)
        min_contour_area: 最小轮廓面积，用于过滤噪声

    Returns:
        List[List[Tuple[int, int]]]: 每个符合要求的红色区域的坐标点列表（原图坐标）
    """
    if image is None or image.size == 0:
        raise ValueError("输入图片为空或无效")

    # 获取图片尺寸
    height, width = image.shape[:2]
    total_area = height * width

    # 裁剪图片，只保留下方66%的区域进行检测
    lower_region_start = int(height * 0.34)  # 下方66%的起始位置
    cropped_image = image[lower_region_start:height, :]  # 裁剪原图
    cropped_height, cropped_width = cropped_image.shape[:2]

    # 将裁剪后的RGB转换为HSV颜色空间
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

    # 创建红色掩码
    # 由于红色在HSV中可能跨越0度边界，需要两个范围
    lower_red1 = np.array(red_lower)
    upper_red1 = np.array(red_upper)

    # 第二个红色范围 (用于跨越0度的红色)
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # 创建两个掩码并合并
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作：开运算去除噪声，闭运算填充空洞
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_regions = []

    # 裁剪区域的面积作为计算基准
    cropped_total_area = cropped_height * cropped_width

    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 计算面积阈值（基于裁剪区域的总面积）
        area_threshold = cropped_total_area * area_threshold_ratio

        # 如果面积大于阈值且大于最小轮廓面积
        if area > area_threshold and area > min_contour_area:
            # 获取轮廓坐标（相对于裁剪区域）
            contour_points = contour.reshape(-1, 2).tolist()

            # 将坐标映射回原图（加上y方向的偏移）
            original_coords = [(point[0], point[1] + lower_region_start) for point in contour_points]

            # 转换为 (x, y) 元组列表
            region_coords = original_coords
            red_regions.append(region_coords)

    return red_regions


def draw_red_regions(image: np.ndarray, regions: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    在图片上绘制检测到的红色区域轮廓

    Args:
        image: 原始RGB图片
        regions: 红色区域坐标列表

    Returns:
        绘制了轮廓的图片副本
    """
    result_image = image.copy()

    for region in regions:
        # 将坐标转换为numpy数组格式用于绘制
        points = np.array(region, dtype=np.int32)
        # 绘制轮廓
        cv2.drawContours(result_image, [points], -1, (0, 255, 0), 2)

    return result_image


def main():
    """
    测试函数
    """
    # 测试图片路径（优先使用创建的测试图片）
    test_images = ['4.jpg', '1.jpg', '2.jpg', '3.jpg', '5.jpg']

    for image_path in test_images:
        try:
            print(f"\n=== 测试图片: {image_path} ===")

            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                continue

            # 转换为RGB格式（OpenCV默认是BGR）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            print(f"图片尺寸: {image_rgb.shape}")

            # 计算图片总面积用于参考
            height, width = image_rgb.shape[:2]
            total_area = height * width

            # 检测红色区域 - 参数说明：
            # red_lower/red_upper: HSV颜色空间中红色的范围，可根据需要调整
            # area_threshold_ratio: 区域面积占整张图片的比例阈值 (0.05 = 5%)
            # min_contour_area: 最小轮廓像素面积，用于过滤噪声
            # 注意：只检测图片下方66%的区域，上方33%的红色区域会被忽略
            red_regions = detect_red_regions(
                image_rgb,
                red_lower=(0, 100, 100),      # 红色下限阈值 (H, S, V) - 更苛刻
                red_upper=(10, 255, 255),   # 红色上限阈值 (H, S, V)
                area_threshold_ratio=0.05,  # 面积阈值 (可调整)
                min_contour_area=3         # 最小轮廓面积 (可调整)
            )

            print(f"检测到 {len(red_regions)} 个红色区域")

            # 显示每个区域的信息
            for i, region in enumerate(red_regions):
                area = cv2.contourArea(np.array(region, dtype=np.int32))
                area_ratio = area / total_area * 100
                print(f"区域 {i+1}: {len(region)} 个坐标点, 面积: {area:.0f}像素 ({area_ratio:.2f}%)")

                # 计算边界框
                if region:
                    x_coords = [point[0] for point in region]
                    y_coords = [point[1] for point in region]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    print(f"  边界框: ({min_x}, {min_y}) 到 ({max_x}, {max_y})")

            # 绘制结果并保存
            result_image = draw_red_regions(image_rgb, red_regions)
            # 转换回BGR保存
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            output_path = f"result_{image_path}"
            cv2.imwrite(output_path, result_bgr)
            print(f"结果已保存到: {output_path}")

        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")


if __name__ == "__main__":
    main()
