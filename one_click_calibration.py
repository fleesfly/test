# -*- coding: utf-8 -*-
"""
一键相机标定工具 v2
参考: OpenCV 官方标定示例 + Halcon 标定流程

核心改进:
  1. 自动采集模式: 检测标定板位置变化，自动捕捉不同视角
  2. 覆盖度引导: 实时显示已采集位置，引导覆盖全画面
  3. 质量预检: 采集前验证清晰度/亮度/对比度
  4. 差图自动剔除: 标定后移除误差最大的图像

流程:
  连接相机 → 自动曝光 → 实时预览 → 移动标定板 → 
  自动采集不同位置 (或按 Space 手动) → 按 C 标定 → 保存

操作:
  SPACE     - 手动采集
  A         - 自动采集模式 (默认)
  C         - 执行标定
  R         - 重置
  D         - 切换覆盖度显示
  Q/ESC     - 退出
"""

import sys
import os
import time
import math
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CALIB_CFG, CAM_CFG, CALIB_IMAGES_DIR, OUTPUT_DIR, CALIB_PARAMS_FILE,
    safe_imwrite
)
from camera_calibration import HalconDotCalibrator
from camera_interface import CameraInterface

# ============================================================
# 配置
# ============================================================

# 自动采集参数
AUTO_CAPTURE = {
    "enabled": True,              # 默认开启自动采集
    "min_translation": 80,        # 标定板中心移动 ≥ 80px 才采集新图
    "min_rotation_change": 15,    # 角度变化 ≥ 15度才采集
    "min_scale_change": 0.15,     # 尺寸变化 ≥ 15% 才采集
    "cooldown_frames": 15,        # 采集后冷却帧数
    "max_images": 30,             # 最大采集数
    "target_images": 15,          # 目标采集数
}

# 覆盖度网格 (将画面分成 3x3 区域)
COVERAGE_GRID = (3, 3)


# ============================================================
# 自动曝光
# ============================================================

def compute_board_position(centers):
    """
    从检测到的圆点中心计算标定板位置特征
    返回: {cx, cy, width, height, spread, angle}
    这些特征用于判断标定板是否移动到了新位置
    """
    if centers is None or len(centers) < 4:
        return None

    c = centers.reshape(-1, 2)
    x, y = c[:, 0], c[:, 1]

    # 中心位置
    cx, cy = float(x.mean()), float(y.mean())

    # 覆盖范围
    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    spread = float(np.sqrt(width**2 + height**2))

    # 用 PCA 估算角度 (第一主成分方向)
    try:
        mean = np.mean(c, axis=0)
        centered = c - mean
        cov = np.cov(centered, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        main_axis = eigenvecs[:, np.argmax(eigenvals)]
        angle = float(np.degrees(np.arctan2(main_axis[1], main_axis[0])))
    except:
        angle = 0.0

    return {
        "cx": cx, "cy": cy,
        "width": width, "height": height,
        "spread": spread, "angle": angle % 180
    }


def is_new_position(new_pos, prev_positions, cfg):
    """
    判断标定板是否移动到了足够新的位置
    与所有已采集位置比较: 中心距离、角度差、尺寸差
    """
    if not prev_positions:
        return True

    for prev in prev_positions:
        # 中心距离
        dist = math.sqrt((new_pos["cx"] - prev["cx"])**2 +
                         (new_pos["cy"] - prev["cy"])**2)
        if dist < cfg["min_translation"]:
            return False

        # 角度变化
        angle_diff = abs(new_pos["angle"] - prev["angle"])
        angle_diff = min(angle_diff, 180 - angle_diff)
        if angle_diff < cfg["min_rotation_change"] and dist < cfg["min_translation"] * 2:
            # 角度变化小且距离不太远，不算新位置
            return False

        # 尺寸变化 (标定板远近)
        if prev["spread"] > 0:
            scale_ratio = new_pos["spread"] / prev["spread"]
            if abs(scale_ratio - 1.0) < cfg["min_scale_change"] and \
               angle_diff < cfg["min_rotation_change"] and \
               dist < cfg["min_translation"] * 1.5:
                return False

    return True


# ============================================================
# 覆盖度可视化
# ============================================================

def update_coverage_grid(coverage_grid, board_pos, image_shape):
    """更新覆盖度网格并返回覆盖比例"""
    h, w = image_shape[:2]
    rows, cols = COVERAGE_GRID
    cell_h, cell_w = h / rows, w / cols

    # 标定板覆盖哪些格子
    if board_pos:
        cx_ratio = board_pos["cx"] / w
        cy_ratio = board_pos["cy"] / h
        col = min(cols - 1, max(0, int(cx_ratio * cols)))
        row = min(rows - 1, max(0, int(cy_ratio * rows)))
        coverage_grid[row][col] = 1

    covered = sum(sum(row) for row in coverage_grid)
    total = rows * cols
    return covered / total


def draw_coverage(display, coverage_grid, image_shape):
    """在预览图上绘制覆盖度热力图"""
    h, w = image_shape[:2]
    rows, cols = COVERAGE_GRID
    cell_h, cell_w = h / rows, w / cols

    overlay = np.zeros_like(display)
    for r in range(rows):
        for c in range(cols):
            if coverage_grid[r][c]:
                y1 = int(r * cell_h)
                x1 = int(c * cell_w)
                y2 = int((r + 1) * cell_h)
                x2 = int((c + 1) * cell_w)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)

    cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)

    # 画网格线
    for r in range(1, rows):
        y = int(r * cell_h)
        cv2.line(display, (0, y), (w, y), (100, 100, 100), 1)
    for c in range(1, cols):
        x = int(c * cell_w)
        cv2.line(display, (x, 0), (x, h), (100, 100, 100), 1)


def draw_captured_positions(display, positions):
    """在预览图上绘制已采集的位置标记"""
    for i, pos in enumerate(positions):
        x, y = int(pos["cx"]), int(pos["cy"])
        cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
        cv2.circle(display, (x, y), 5, (255, 255, 255), 1)
        cv2.putText(display, str(i + 1), (x + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# ============================================================
# 差图剔除
# ============================================================

def remove_outlier_images(calibrator, max_remove_ratio=0.2):
    """剔除重投影误差最大的图像"""
    if not hasattr(calibrator, '_per_view_errors') or not calibrator._per_view_errors:
        return 0

    errors = calibrator._per_view_errors.copy()
    num_to_remove = max(0, min(
        int(len(errors) * max_remove_ratio),
        len(errors) - CALIB_CFG.min_calib_images
    ))
    if num_to_remove == 0:
        return 0

    worst_indices = np.argsort(errors)[-num_to_remove:][::-1]
    removed = 0
    for idx in sorted(worst_indices, reverse=True):
        calibrator.object_points.pop(idx)
        calibrator.image_points.pop(idx)
        calibrator.image_shapes.pop(idx)
        if idx < len(calibrator._calib_frames):
            calibrator._calib_frames.pop(idx)
        if idx < len(calibrator._calib_annotated):
            calibrator._calib_annotated.pop(idx)
        if idx < len(calibrator._calib_detected_centers):
            calibrator._calib_detected_centers.pop(idx)
        removed += 1

    print(f"[标定] 已剔除 {removed} 张差图")
    return removed


# ============================================================
# 采集建议生成
# ============================================================

def get_capture_suggestion(coverage_grid):
    """根据覆盖度给出下一步采集建议"""
    rows, cols = COVERAGE_GRID
    uncovered = []
    for r in range(rows):
        for c in range(cols):
            if not coverage_grid[r][c]:
                uncovered.append((r, c))

    if not uncovered:
        return "覆盖度良好! 可以按 C 标定了"

    # 建议去未覆盖的区域
    regions = {
        (0, 0): "左上角", (0, 1): "上方中间", (0, 2): "右上角",
        (1, 0): "左中", (1, 1): "正中心", (1, 2): "右中",
        (2, 0): "左下角", (2, 1): "下方中间", (2, 2): "右下角",
    }

    suggestions = []
    for r, c in uncovered[:3]:
        reg = regions.get((r, c), f"({r},{c})")
        suggestions.append(reg)

    return f"建议移动标定板到: {'/'.join(suggestions)}"


# ============================================================
# 标定结果保存
# ============================================================

def save_calibration_results(calibrator):
    """保存标定结果到 output 文件夹"""
    info = calibrator.evaluate_calibration()
    rms = info["rms_error_px"]

    # 保存参数
    calibrator.save()

    # 预览图
    pre_dir = os.path.join(OUTPUT_DIR, "calib_preview")
    os.makedirs(pre_dir, exist_ok=True)
    for i in range(calibrator.num_captured):
        ann = calibrator.get_annotated_image(i)
        if ann is not None:
            safe_imwrite(os.path.join(pre_dir, f"preview_{i+1:02d}_detected.png"), ann)
        raw = calibrator.get_captured_image(i)
        if raw is not None:
            safe_imwrite(os.path.join(OUTPUT_DIR, f"calib_{i+1:02d}_original.png"), raw)
        ann2 = calibrator.get_annotated_image(i)
        if ann2 is not None:
            safe_imwrite(os.path.join(OUTPUT_DIR, f"calib_{i+1:02d}_detected.png"), ann2)

    # 结果摘要图
    result_img = np.ones((400, 750, 3), dtype=np.uint8) * 30
    lines = [
        "Camera Calibration Result",
        "",
        f"Images used: {calibrator.num_captured}",
        f"RMS Error: {rms:.4f} px",
        f"RMS Quality: {info['rms_quality']}",
        f"Focal: fx={info['focal_length'][0]:.1f}, fy={info['focal_length'][1]:.1f}",
        f"Principal: cx={info['principal_point'][0]:.1f}, cy={info['principal_point'][1]:.1f}",
        "",
        "Distortion Coefficients:",
        f"  k1 = {info['distortion_coefficients']['k1 (径向)']:.6f}",
        f"  k2 = {info['distortion_coefficients']['k2 (径向)']:.6f}",
        f"  p1 = {info['distortion_coefficients']['p1 (切向)']:.6f}",
        f"  p2 = {info['distortion_coefficients']['p2 (切向)']:.6f}",
        f"  k3 = {info['distortion_coefficients']['k3 (径向)']:.6f}",
    ]
    for i, line in enumerate(lines):
        y = 25 + i * 26
        cv2.putText(result_img, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
    safe_imwrite(os.path.join(OUTPUT_DIR, "calibration_result.png"), result_img)

    print(f"\n  {'='*55}")
    print(f"  标定完成!")
    print(f"  RMS: {rms:.4f} px | 评级: {info['rms_quality']}")
    print(f"  焦距: fx={info['focal_length'][0]:.1f}, fy={info['focal_length'][1]:.1f}")
    print(f"  主点: cx={info['principal_point'][0]:.1f}, cy={info['principal_point'][1]:.1f}")
    print(f"")
    print(f"  保存文件:")
    print(f"  - config/calibration_params.json")
    print(f"  - output/calib_preview/*.png")
    print(f"  - output/calib_*_original.png")
    print(f"  - output/calib_*_detected.png")
    print(f"  - output/calibration_result.png")
    print(f"  {'='*55}")
    return info


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 55)
    print("  一键相机标定 v2")
    print("  Halcon 7x7 圆点阵 | 张正友标定法")
    print("  参考: OpenCV 标定示例 + 自动采集策略")
    print("=" * 55)

    calibrator = HalconDotCalibrator()
    camera = None

    try:
        # [1] 连接相机
        print("\n[1/5] 连接相机...")
        camera = CameraInterface()
        if not camera.open():
            print("\n!" * 45)
            print("  相机连接失败! 请检查:")
            print("  1. USB线是否连接    2. Vimba SDK是否安装")
            print("  3. Vimba Viewer是否占用  4. 重新插拔USB")
            print("!" * 45)
            return False

        # [2] 启动采集
        print("\n[2/5] 启动采集...")
        camera.start_streaming(None)

        frame = None
        for _ in range(20):
            frame = camera.get_frame(timeout=1.0)
            if frame is not None:
                break
            time.sleep(0.1)
        if frame is None:
            print("[错误] 无法获取图像")
            return False

        h, w = frame.shape[:2]
        print(f"      图像: {w}x{h}")

        print("\n[4/5] 采集标定图像")
        print("  " + "-" * 50)
        print("  操作方法:")
        print("  移动标定板 → 系统自动采集不同位置")
        print("  [Space] 手动采集  [A]切换自动/手动")
        print("  [C]标定  [R]重置  [D]覆盖图  [Q]退出")
        print("  " + "-" * 50)
        print(f"  目标: {AUTO_CAPTURE['target_images']} 张, "
              f"覆盖 {COVERAGE_GRID[0]}x{COVERAGE_GRID[1]} 区域")
        print()

        # 状态变量
        captured_count = 0
        auto_mode = AUTO_CAPTURE["enabled"]
        show_coverage = True
        running = True
        last_msg = ""
        last_msg_time = 0
        cooldown = 0
        calib_done = False

        # 已采集位置列表 (用于判断新位置)
        prev_positions = []
        # 覆盖度网格
        coverage_grid = [[0] * COVERAGE_GRID[1] for _ in range(COVERAGE_GRID[0])]

        cv2.namedWindow("一键标定", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("一键标定", 960, 720)

        while running:
            frame = camera.get_frame(timeout=0.5)
            if frame is None:
                continue

            # 转灰度
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display = frame.copy()
            else:
                gray = frame.copy()
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # 检测圆点
            success, centers, annotated = calibrator.detect_dots(gray)

            # 计算标定板位置
            board_pos = compute_board_position(centers) if success else None

            # 更新覆盖度
            if board_pos:
                cov_ratio = update_coverage_grid(coverage_grid, board_pos, (h, w))
            else:
                cov_ratio = sum(sum(r) for r in coverage_grid) / (COVERAGE_GRID[0] * COVERAGE_GRID[1])

            # ---- 顶栏信息 ----
            mode_text = "AUTO" if auto_mode else "MANUAL"
            mode_color = (0, 255, 0) if auto_mode else (255, 255, 0)
            cv2.putText(display, f"[{mode_text}] 采集: {captured_count}/{AUTO_CAPTURE['target_images']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            cv2.putText(display, f"覆盖: {int(cov_ratio*100)}%",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1)

            if success:
                cv2.putText(display, "DETECTED", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "NO BOARD", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if calib_done:
                cv2.putText(display, "CALIBRATED", (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # ---- 覆盖度热力图 ----
            if show_coverage:
                draw_coverage(display, coverage_grid, (h, w))

            # ---- 已采集位置标记 ----
            if prev_positions:
                draw_captured_positions(display, prev_positions)

            # ---- 圆点检测叠加 ----
            if success and annotated is not None:
                display = annotated

            # ---- 提示 ----
            if last_msg and time.time() - last_msg_time < 3.0:
                cv2.putText(display, last_msg, (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # ---- 建议 ----
            suggestion = get_capture_suggestion(coverage_grid)
            cv2.putText(display, suggestion, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("一键标定", display)
            key = cv2.waitKey(30) & 0xFF

            # ---- 冷却递减 ----
            if cooldown > 0:
                cooldown -= 1

            # ---- 自动采集逻辑 ----
            if auto_mode and success and board_pos and cooldown == 0 and \
               captured_count < AUTO_CAPTURE["max_images"]:
                if is_new_position(board_pos, prev_positions, AUTO_CAPTURE):
                    # 质量验证
                    quality = calibrator.validate_image_quality(gray)
                    if quality["valid"] or len(quality["warnings"]) <= 1:
                        _do_capture(calibrator, gray, centers, annotated,
                                    display, board_pos, prev_positions,
                                    captured_count, coverage_grid, (h, w))
                        captured_count += 1
                        cooldown = AUTO_CAPTURE["cooldown_frames"]
                        last_msg = f"自动采集 #{captured_count}"
                        last_msg_time = time.time()
                        print(f"  [自动 #{captured_count:02d}] "
                              f"位置=({board_pos['cx']:.0f},{board_pos['cy']:.0f}) "
                              f"角度={board_pos['angle']:.0f}°")

            # ---- 按键处理 ----
            if key == 27 or key == ord("q"):
                print("  用户退出")
                running = False

            elif key == ord(" "):
                # 手动采集
                if not success or board_pos is None:
                    last_msg = "未检测到标定板"
                    last_msg_time = time.time()
                    continue

                quality = calibrator.validate_image_quality(gray)
                if not quality["valid"]:
                    last_msg = f"质量不合格: {quality['warnings'][0]}"
                    last_msg_time = time.time()
                    continue

                _do_capture(calibrator, gray, centers, annotated,
                            display, board_pos, prev_positions,
                            captured_count, coverage_grid, (h, w))
                captured_count += 1
                last_msg = f"手动采集 #{captured_count}"
                last_msg_time = time.time()
                print(f"  [手动 #{captured_count:02d}] 采集成功")

            elif key == ord("a"):
                auto_mode = not auto_mode
                print(f"  {'自动' if auto_mode else '手动'}模式")

            elif key == ord("c"):
                if calibrator.num_captured < CALIB_CFG.min_calib_images:
                    msg = f"需要≥{CALIB_CFG.min_calib_images}张, 当前{calibrator.num_captured}"
                    print(f"  [标定] {msg}")
                    last_msg = msg
                    last_msg_time = time.time()
                    continue

                print(f"\n[5/5] 标定 ({calibrator.num_captured}张)...")
                ok = calibrator.calibrate()
                if ok:
                    removed = remove_outlier_images(calibrator)
                    if removed > 0:
                        print(f"  重标定 ({calibrator.num_captured}张)...")
                        ok = calibrator.calibrate()

                if ok:
                    calib_done = True
                    info = save_calibration_results(calibrator)
                    last_msg = f"OK! RMS={info['rms_error_px']:.4f}px"
                else:
                    last_msg = "标定失败"
                last_msg_time = time.time()

            elif key == ord("r"):
                calibrator.reset()
                captured_count = 0
                prev_positions.clear()
                coverage_grid = [[0] * COVERAGE_GRID[1] for _ in range(COVERAGE_GRID[0])]
                calib_done = False
                print("  已重置")

            elif key == ord("d"):
                show_coverage = not show_coverage

    finally:
        cv2.destroyAllWindows()
        if camera:
            try:
                camera.stop_streaming()
            except:
                pass
            try:
                camera.close()
            except:
                pass
        print("\n退出")

    return True


def _do_capture(calibrator, gray, centers, annotated, display,
                board_pos, prev_positions, captured_count,
                coverage_grid, image_shape):
    """执行图像采集"""
    calibrator.add_calibration_image(gray, centers, annotated)

    ts = datetime.now().strftime("%H%M%S_%f")[:-3]
    idx = captured_count + 1

    raw_path = os.path.join(CALIB_IMAGES_DIR, f"calib_{idx:02d}_{ts}_original.png")
    det_path = os.path.join(CALIB_IMAGES_DIR, f"calib_{idx:02d}_{ts}_detected.png")
    safe_imwrite(raw_path, gray)
    safe_imwrite(det_path, annotated)

    # 记录位置
    prev_positions.append(board_pos)
    update_coverage_grid(coverage_grid, board_pos, image_shape)


if __name__ == "__main__":
    main()
