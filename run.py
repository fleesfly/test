# -*- coding: utf-8 -*-
"""
金属板表面划痕实时检测系统 - 启动脚本

使用方式:
    python run.py              # 启动完整 GUI
    python run.py --test       # 运行算法自测 (无需相机)
    python run.py --calib-only # 仅运行标定模块测试
"""

import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_test():
    """运行划痕检测算法自测"""
    print("=" * 60)
    print("  金属板划痕检测算法自测")
    print("=" * 60)

    import cv2
    import numpy as np
    from config import SCRATCH_CFG, OUTPUT_DIR, safe_imwrite
    from scratch_detector import ScratchDetector

    # 生成测试图像
    print("\n[1] 生成模拟金属板测试图像...")
    test_img = np.ones((600, 800), dtype=np.uint8) * 160
    noise = np.random.normal(0, 6, test_img.shape).astype(np.int16)
    test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 多种划痕类型
    cv2.line(test_img, (100, 100), (450, 115), 30, 2)     # 长划痕
    cv2.line(test_img, (200, 200), (380, 210), 20, 1)     # 细划痕
    cv2.line(test_img, (550, 80), (570, 480), 45, 3)      # 垂直粗划痕
    cv2.line(test_img, (50, 350), (350, 365), 120, 1)     # 亮划痕
    cv2.line(test_img, (400, 400), (650, 405), 70, 2)     # 倾斜划痕
    cv2.line(test_img, (500, 500), (700, 520), 80, 1)     # 短细划痕

    # 保存测试原图
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_imwrite(os.path.join(OUTPUT_DIR, "test_original.png"), test_img)
    print("  测试图像已保存: output/test_original.png")

    # 运行检测
    print("\n[2] 运行划痕检测...")
    detector = ScratchDetector()
    scratches, mask = detector.detect(test_img)

    print(f"\n  检测到 {len(scratches)} 条划痕:")
    print(f"  {'ID':<4} {'长度px':<9} {'宽度px':<9} {'角度°':<8} {'面积':<9} {'长宽比':<7} {'置信度':<7}")
    print(f"  {'─'*55}")
    for s in scratches:
        print(f"  #{s.id:<3} {s.length_px:<9.1f} {s.width_px:<9.1f} "
              f"{s.angle_deg:<8.1f} {s.area_px2:<9.0f} "
              f"{s.aspect_ratio:<7.2f} {s.confidence:<7.2f}")

    # 保存结果可视化
    result = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    palette = [
        (0, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 165, 0), (0, 128, 255), (128, 0, 255),
    ]
    for s in scratches:
        color = palette[(s.id - 1) % len(palette)]
        cv2.drawContours(result, [s.contour], -1, color, 2)
        cx, cy = int(s.centroid[0]), int(s.centroid[1])
        cv2.putText(result, f"#{s.id} L={s.length_px:.0f}",
                    (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = os.path.join(OUTPUT_DIR, "test_result.png")
    safe_imwrite(out_path, result)
    print(f"\n[3] 结果图像已保存: {out_path}")

    # 掩膜
    mask_path = os.path.join(OUTPUT_DIR, "test_mask.png")
    safe_imwrite(mask_path, mask)
    print(f"  缺陷掩膜已保存: {mask_path}")

    print("\n" + "=" * 60)
    print("  自测完成!")
    print("=" * 60)
    return True


def run_calib_test():
    """测试标定模块"""
    print("=" * 60)
    print("  标定模块自测")
    print("=" * 60)

    from camera_calibration import HalconDotCalibrator

    calibrator = HalconDotCalibrator()

    if calibrator.load():
        import json
        print("✓ 已加载标定参数:")
        print(json.dumps(calibrator.evaluate_calibration(),
                         indent=2, ensure_ascii=False))
    else:
        print("✗ 未找到标定参数文件")
        print("  请使用 GUI 采集标定图像并执行标定")


def main():
    """主入口"""
    args = sys.argv[1:]

    if "--test" in args or "-t" in args:
        run_test()
    elif "--calib-only" in args:
        run_calib_test()
    elif "--help" in args or "-h" in args:
        print(__doc__)
        print("选项:")
        print("  --test, -t       运行算法自测")
        print("  --calib-only     仅测试标定模块")
        print("  --help, -h       显示帮助")
    else:
        # 启动 GUI
        from main_app import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()

