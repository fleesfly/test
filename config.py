# -*- coding: utf-8 -*-
"""
金属板表面划痕检测系统 - 配置文件
Hardware: Allied Vision Mako/Goldeye/Alvium + Vimba SDK
Calibration: Halcon-style 7x7 dot grid, 10mm spacing
"""

import os
import json
import cv2
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_IMAGES_DIR = os.path.join(BASE_DIR, "calib_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CALIB_PARAMS_FILE = os.path.join(BASE_DIR, "config", "calibration_params.json")

os.makedirs(CALIB_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 标定板配置: Halcon 7x7 圆点阵
# ============================================================
@dataclass
class CalibrationConfig:
    """Halcon 标定板参数"""
    dot_grid_rows: int = 7
    dot_grid_cols: int = 7
    dot_spacing_mm: float = 10.0               # 点间距 (mm)
    min_calib_images: int = 10                  # 最少标定图像数
    max_calib_images: int = 30                  # 最多标定图像数

    # 圆点检测参数 (findCirclesGrid Blob Detector)
    blob_filter_by_color: bool = True
    blob_color: int = 0                         # 0=黑点, 255=白点
    blob_min_area: float = 50.0                 # 最小圆点面积 (px²)
    blob_max_area: float = 5000.0               # 最大圆点面积 (px²)
    blob_min_circularity: float = 0.7           # 最小圆形度

    # 圆点检测参数 (HoughCircles Fallback)
    hough_dp: float = 1.0
    hough_min_dist: int = 20
    hough_param1: int = 30
    hough_param2: int = 15
    min_radius: int = 3
    max_radius: int = 50

    # 亚像素优化
    subpix_win_size: Tuple[int, int] = (11, 11)
    subpix_zero_zone: Tuple[int, int] = (-1, -1)
    subpix_max_iter: int = 30
    subpix_epsilon: float = 0.001

    # 标定标志 (张正友标定法: 仅使用初始猜测, 让优化自由求解)
    # 不固定主点和畸变系数, 使标定结果更准确
    calib_flags: int = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_RATIONAL_MODEL      # k3 径向畸变
        | cv2.CALIB_THIN_PRISM_MODEL    # 薄棱镜畸变
    )


# ============================================================
# 相机配置
# ============================================================
@dataclass
class CameraConfig:
    """Allied Vision 相机参数"""
    # --- 采集参数 ---
    exposure_time_us: float = 50000.0            # 曝光时间 (微秒)
    gain_db: float = 0.0                        # 增益 (dB)
    frame_rate_hz: float = 30.0                 # 帧率
    pixel_format: str = "Mono8"                 # 像素格式
    width: int = 640                              # 0=使用相机最大分辨率
    height: int = 480
    offset_x: int = 0
    offset_y: int = 0

    # --- 触发模式 ---
    trigger_source: str = "Freerun"             # Freerun / Line1 / Software

    # --- 预处理 ---
    enable_gamma: bool = False
    gamma_value: float = 1.0

    # --- ROI ---
    use_roi: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 0
    roi_h: int = 0


# ============================================================
# 划痕检测配置
# ============================================================
@dataclass
class ScratchDetectionConfig:
    """划痕检测算法参数"""
    # --- 预处理 ---
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    enable_bilateral: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # --- 边缘检测 ---
    canny_low: int = 60
    canny_high: int = 180
    canny_aperture: int = 3

    # --- 形态学 ---
    morph_kernel_size: int = 3
    morph_close_iterations: int = 3
    morph_open_iterations: int = 2

    # --- 线段检测 ---
    hough_rho: float = 1.0
    hough_theta: float = 1.0 * 3.14159 / 180
    hough_threshold: int = 30
    hough_min_line_length: float = 30.0
    hough_max_line_gap: float = 10.0

    # --- Top-Hat 检测 (细划痕) ---
    enable_tophat: bool = True
    tophat_kernel_size: int = 15

    # --- 频域分析 ---
    enable_fft: bool = False
    fft_threshold: float = 0.15

    # --- Gabor 滤波 ---
    enable_gabor: bool = True
    gabor_ksize: int = 21
    gabor_sigma: float = 4.0
    gabor_theta: float = 0.0
    gabor_lambd: float = 10.0
    gabor_gamma: float = 0.5
    gabor_psi: float = 0.0

    # --- 划痕判定阈值 ---
    min_scratch_length_px: float = 25.0         # 最小划痕长度 (像素)
    min_scratch_area_px2: float = 100.0          # 最小划痕面积 (像素²)
    max_scratch_width_px: float = 50.0          # 最大划痕宽度 (像素)
    scratch_aspect_ratio_min: float = 3.0       # 最小长宽比 (区分划痕与点状缺陷)

    # --- 多尺度检测 ---
    enable_multiscale: bool = True
    scales: tuple = (0.8, 1.0, 1.2)

    # --- code1 适配参数 (Canny + 轮廓分析法) ---
    gamma: float = 1.2                       # 伽马校正值
    max_scratch_count: int = 30              # 最大保留轮廓数
    min_contour_length: float = 60.0         # 轮廓最小周长
    filter_enable: bool = True               # 是否启用误检过滤
    area_threshold: float = 60.0             # 有效划痕最小面积


# ============================================================
# GUI 配置
# ============================================================
@dataclass
class GUIConfig:
    """GUI 界面参数"""
    window_title: str = "金属板表面划痕实时检测系统"
    window_width: int = 1920
    window_height: int = 1080
    preview_update_ms: int = 60                   # 预览刷新间隔 (ms)
    max_display_width: int = 1280
    max_display_height: int = 960
    overlay_color: Tuple[int, int, int] = (0, 255, 0)   # 标记颜色 (BGR)
    overlay_thickness: int = 2
    font_scale: float = 0.6
    info_max_lines: int = 20


# ============================================================
# 默认配置实例
# ============================================================
CALIB_CFG = CalibrationConfig()
CAM_CFG = CameraConfig()
SCRATCH_CFG = ScratchDetectionConfig()
GUI_CFG = GUIConfig()

# ============================================================
# 配置保存/加载
# ============================================================
def save_calibration_params(camera_matrix, dist_coeffs, rms_error,
                            image_size, filepath=CALIB_PARAMS_FILE,
                            extrinsics=None):
    """保存标定参数到 JSON (兼容中文路径)"""
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rms_error": float(rms_error),
        "image_size": list(image_size),
        "method": "张正友标定法 (Zhang's Method)",
        "description": "Halcon 7x7 dot pattern calibration",
        "extrinsics": extrinsics or [],
    }
    safe_json_save(filepath, data)
    return filepath


def load_calibration_params(filepath=CALIB_PARAMS_FILE):
    """加载标定参数 (兼容中文路径)"""
    data = safe_json_load(filepath)
    if data is None:
        return None
    return {
        "camera_matrix": np.array(data["camera_matrix"]),
        "dist_coeffs": np.array(data["dist_coeffs"]),
        "rms_error": data["rms_error"],
        "image_size": tuple(data["image_size"])
    }


# 延迟导入 numpy
import numpy as np
# ============================================================
# 中文路径安全写入工具
# ============================================================

def safe_imwrite(filepath, image, params=None):
    """
    安全写入图像文件 (兼容中文路径)

    Windows 下 OpenCV 的 cv2.imwrite 对非 ASCII 路径支持有限，
    使用 imencode + tofile 绕过编码问题。

    Args:
        filepath: 图像文件路径
        image: numpy 图像数组
        params: 编码参数 (可选)

    Returns:
        bool: 写入成功返回 True
    """
    import cv2 as _cv2
    import warnings
    warnings.filterwarnings("ignore", message=".*iCCP.*sRGB.*")
    ext = os.path.splitext(filepath)[1].lower()
    if not ext:
        ext = '.png'
        filepath += ext
    # imencode 默认 PNG 压缩参数
    encode_params = params or [_cv2.IMWRITE_PNG_COMPRESSION, 3]
    ret, buf = _cv2.imencode(ext, image, encode_params)
    if ret:
        buf.tofile(filepath)
    return ret


def safe_json_save(filepath, data):
    """
    安全写入 JSON 文件 (兼容中文路径)

    显式使用 utf-8 编码，确保中文字符正确存储。
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_json_load(filepath):
    """
    安全读取 JSON 文件 (兼容中文路径)
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
