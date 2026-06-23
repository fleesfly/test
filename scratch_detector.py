# -*- coding: utf-8 -*-
"""
金属板表面划痕检测核心算法模块

检测策略（多方法融合）:
  1. Top-Hat 形态学      → 细划痕（对比度低、宽度窄）
  2. Canny + 概率霍夫变换 → 长划痕（边缘清晰）
  3. Gabor 方向滤波器     → 方向性纹理划痕
  4. 自适应阈值分割       → 深浅不一的大面积划痕
  5. 多尺度融合           → 不同宽度的划痕

后处理:
  - 连通域分析 → 分离每个划痕
  - 几何特征计算 → 长度、宽度、角度、面积、长宽比
  - 伪缺陷过滤 → 根据几何特征剔除噪声
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from config import SCRATCH_CFG


@dataclass
class ScratchInfo:
    """单个划痕信息"""
    id: int
    contour: np.ndarray
    length_px: float
    width_px: float
    area_px2: float
    angle_deg: float             # 主方向角度
    aspect_ratio: float           # 长宽比
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    centroid: Tuple[float, float] = (0.0, 0.0)  # 质心
    points: np.ndarray = field(default_factory=lambda: np.array([]))  # 骨架点
    confidence: float = 0.0             # 置信度 (0~1)
    false_alarm: bool = False       # 是否标记为误检


class ScratchDetector:
    """
    金属板表面划痕检测器

    融合多种算法，适用于:
      - 镜面金属板 (高反射)
      - 拉丝金属板 (方向性纹理)
      - 哑光金属板 (漫反射)
    """

    def __init__(self, config=None, camera_matrix=None, dist_coeffs=None):
        self.cfg = config or SCRATCH_CFG
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # 预初始化 Gabor 核 (性能优化)
        self._gabor_kernels: List[Tuple[float, np.ndarray]] = []
        self._init_gabor_kernels()

        # CLAHE
        self._clahe = (cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=(self.cfg.clahe_tile_size, self.cfg.clahe_tile_size)
        ) if self.cfg.enable_clahe else None)

    # ================================================================
    # 主入口
    # ================================================================

    def detect(self, image: np.ndarray) -> Tuple[List[ScratchInfo], np.ndarray]:
        """
        检测图像中的所有划痕 (Canny + 轮廓分析法，适配自 code1)

        Args:
            image: 输入图像 (BGR 或 灰度)

        Returns:
            (scratches, defect_mask)
            - scratches: 划痕列表
            - defect_mask: 二值缺陷掩膜
        """
        # 转灰度
        gray = self._to_gray(image)

        # 1. 伽马校正
        gamma = self.cfg.gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(gray, table)

        # 2. 中值滤波去噪
        denoised = cv2.medianBlur(corrected, 5)

        # 3. CLAHE 增强
        if self.cfg.enable_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.cfg.clahe_clip_limit,
                                     tileGridSize=(self.cfg.clahe_tile_size, self.cfg.clahe_tile_size))
            enhanced = clahe.apply(denoised)
        else:
            enhanced = denoised

        # 4. Canny 边缘检测
        low = self.cfg.canny_low
        high = self.cfg.canny_high
        edges = cv2.Canny(enhanced, low, high)

        # 5. 形态学闭运算连接断裂边缘
        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 6. 轮廓提取
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7. 按周长排序取前 N 个
        max_keep = self.cfg.max_scratch_count
        min_len = self.cfg.min_contour_length
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:max_keep]

        # 8. 过滤与构建 ScratchInfo
        scratches = []
        mask_h, mask_w = gray.shape
        defect_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        filter_on = self.cfg.filter_enable
        area_thresh = self.cfg.area_threshold
        ar_min = self.cfg.scratch_aspect_ratio_min

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < min_len:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # 计算几何特征（minAreaRect 提供真实长宽比）
            length = float(peri)
            width_px = float(area / max(peri, 1) * 2) if peri > 0 else 0.0
            angle = 0.0
            real_ar = 1.0  # 真实长宽比
            if len(cnt) >= 5:
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
                angle = float(ang)
                length = max(rw, rh)
                width_px = min(rw, rh)
                real_ar = length / max(width_px, 1.0)

            # 误检判断（用 minAreaRect 的真实长宽比，不用 boundingRect）
            is_false = False
            if filter_on:
                if peri < min_len * 1.5 and area < area_thresh:
                    is_false = True
                elif real_ar < ar_min:
                    is_false = True

            confidence = 0.3 if is_false else 0.9
            aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

            sid = len(scratches) + 1
            s = ScratchInfo(
                id=sid,
                bbox=(x, y, w, h),
                contour=cnt,
                length_px=length,
                width_px=width_px,
                angle_deg=angle,
                area_px2=area,
                aspect_ratio=aspect,
                confidence=confidence,
                false_alarm=is_false,
            )
            scratches.append(s)
            cv2.drawContours(defect_mask, [cnt], -1, 255, -1)

        return scratches, defect_mask

    # ================================================================
    # 预处理
    # ================================================================

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """安全转灰度: 兼容 (H,W), (H,W,1), (H,W,3)"""
        if len(image.shape) == 2:
            return image.copy()
        if image.shape[2] == 1:
            return image[:, :, 0].copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """
        图像预处理管线

        1. 畸变校正 (如果有标定参数)
        2. 双边滤波 (去噪保边)
        3. CLAHE 对比度增强
        """
        # 畸变校正
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = gray.shape[:2]
            new_mtx, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
            gray = cv2.undistort(gray, self.camera_matrix, self.dist_coeffs,
                                  None, new_mtx)

        # 双边滤波
        if self.cfg.enable_bilateral:
            # 减弱平滑强度以保留细划痕
            gray = cv2.bilateralFilter(
                gray, self.cfg.bilateral_d,
                self.cfg.bilateral_sigma_color,
                self.cfg.bilateral_sigma_space
            )

        # CLAHE
        if self.cfg.enable_clahe and self._clahe is not None:
            gray = self._clahe.apply(gray)

        return gray

    # ================================================================
    # 方法1: Top-Hat 形态学 (细划痕)
    # ================================================================

    def _detect_tophat(self, gray: np.ndarray) -> np.ndarray:
        """
        Top-Hat (顶帽) 变换检测细划痕

        原理: 原图 - 开运算(原图) = 比周围亮的细小结构
        对于暗划痕: Black-Hat = 闭运算(原图) - 原图
        """
        if not self.cfg.enable_tophat:
            return np.zeros_like(gray, dtype=np.uint8)

        ks = self.cfg.tophat_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        # 同时检测亮划痕和暗划痕
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        combined = cv2.addWeighted(tophat, 0.5, blackhat, 0.5, 0)

        # OTSU 自适应阈值
        _, binary = cv2.threshold(
            combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    # ================================================================
    # 方法2: Canny + 概率霍夫变换 (长划痕)
    # ================================================================

    def _detect_canny_hough(self, gray: np.ndarray) -> np.ndarray:
        """
        Canny 边缘 + 概率霍夫变换检测长划痕

        适用于: 对比度高、边缘清晰的长划痕
        """
        # Canny 边缘检测
        edges = cv2.Canny(
            gray,
            self.cfg.canny_low,
            self.cfg.canny_high,
            apertureSize=self.cfg.canny_aperture
        )

        # 概率霍夫变换
        # 降低阈值检测更多短线
        lines = cv2.HoughLinesP(
            edges,
            self.cfg.hough_rho,
            self.cfg.hough_theta,
            self.cfg.hough_threshold,
            minLineLength=self.cfg.hough_min_line_length,
            maxLineGap=self.cfg.hough_max_line_gap
        )

        # 在空白画布上绘制线段
        line_mask = np.zeros_like(gray, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        # 膨胀连接断点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_mask = cv2.dilate(line_mask, kernel, iterations=1)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return line_mask

    # ================================================================
    # 方法3: Gabor 方向滤波器
    # ================================================================

    def _init_gabor_kernels(self):
        """预初始化多方向 Gabor 核"""
        if not self.cfg.enable_gabor:
            return
        thetas = np.linspace(0, np.pi, 12, endpoint=False)  # 12 方向全覆盖
        g = self.cfg
        for theta in thetas:
            kernel = cv2.getGaborKernel(
                (g.gabor_ksize, g.gabor_ksize),
                g.gabor_sigma, theta,
                g.gabor_lambd, g.gabor_gamma, g.gabor_psi,
                ktype=cv2.CV_32F
            )
            self._gabor_kernels.append((theta, kernel))

    def _detect_gabor(self, gray: np.ndarray) -> np.ndarray:
        """
        Gabor 滤波器组检测方向性划痕

        多方向的 Gabor 滤波，取最大响应，阈值化后得到划痕区域
        """
        if not self.cfg.enable_gabor or not self._gabor_kernels:
            return np.zeros_like(gray, dtype=np.uint8)

        gray_f = gray.astype(np.float32) / 255.0

        max_response = np.zeros_like(gray_f)
        for theta, kernel in self._gabor_kernels:
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            max_response = np.maximum(max_response, np.abs(filtered))

        # 归一化并阈值化
        max_response = (max_response * 255).astype(np.uint8)
        _, binary = cv2.threshold(
            max_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return binary

    # ================================================================
    # 掩膜融合
    # ================================================================

    def _fuse_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        融合多个检测掩膜

        策略: 或操作 + 形态学闭合
        """
        fused = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            fused = cv2.bitwise_or(fused, mask)

        # 形态学闭合，连接断点
        ks = self.cfg.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel,
                                  iterations=self.cfg.morph_close_iterations)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel,
                                  iterations=self.cfg.morph_open_iterations)
        return fused

    # ================================================================
    # 划痕提取与分析
    # ================================================================

    def _extract_scratches(self, defect_mask: np.ndarray
                           ) -> List[ScratchInfo]:
        """
        从缺陷掩膜中提取每个划痕的几何信息

        处理流程:
          连通域分析 → 几何特征计算 → 伪缺陷过滤 → 排序
        """
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            defect_mask, connectivity=8
        )

        scratches: List[ScratchInfo] = []
        scratch_id = 0

        for i in range(1, num_labels):  # 跳过背景 (label 0)
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.min_scratch_area_px2:
                continue

            # 提取该连通域的轮廓
            region_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)

            # 计算几何特征
            length, width, angle = self._compute_scratch_geometry(contour, region_mask)
            aspect_ratio = length / max(width, 1.0)

            # 过滤伪缺陷
            if length < self.cfg.min_scratch_length_px:
                continue
            if width > self.cfg.max_scratch_width_px:
                continue
            if aspect_ratio < self.cfg.scratch_aspect_ratio_min:
                continue

            scratch_id += 1
            cx, cy = centroids[i]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            # 骨架
            skeleton = self._get_skeleton_points(region_mask)

            scratches.append(ScratchInfo(
                id=scratch_id,
                contour=contour,
                length_px=length,
                width_px=width,
                area_px2=area,
                angle_deg=angle,
                aspect_ratio=aspect_ratio,
                centroid=(float(cx), float(cy)),
                bbox=(x, y, w, h),
                points=skeleton,
                confidence=self._compute_confidence(length, width, area, aspect_ratio)
            ))

        # 按划痕长度降序排列
        scratches.sort(key=lambda s: s.length_px, reverse=True)
        # 重新编号
        for idx, s in enumerate(scratches):
            s.id = idx + 1

        return scratches

    def _compute_scratch_geometry(self, contour: np.ndarray,
                                   region_mask: np.ndarray
                                   ) -> Tuple[float, float, float]:
        """
        计算划痕的长度、宽度和主方向角度

        方法:
          1. 使用轮廓的最小外接矩形获取角度和近似尺寸
          2. 使用距离变换获取更准确的宽度
        """
        if len(contour) < 5:
            return 0.0, 0.0, 0.0

        # 最小外接矩形 (拟合)
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect

        # 长度 = 长边, 宽度 = 短边
        length = max(rw, rh)
        width_mbr = min(rw, rh)

        # 距离变换获取更准确的宽度
        dist = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
        valid_dist = dist[dist > 0]
        if len(valid_dist) > 10:
            # 宽度 ≈ 2 × 距离变换中位数 (划痕中线到边缘)
            width_dt = 2.0 * float(np.median(valid_dist))
        else:
            width_dt = width_mbr

        width = min(width_mbr, width_dt)  # 取更保守的宽度

        return length, width, angle

    def _get_skeleton_points(self, region_mask: np.ndarray) -> np.ndarray:
        """提取划痕骨架点 (用于可视化)"""
        # 细化算法
        skeleton = cv2.ximgproc.thinning(region_mask) if hasattr(
            cv2, "ximgproc") else self._simple_thin(region_mask)
        pts = np.column_stack(np.where(skeleton > 0))
        return pts[:, ::-1].astype(np.float32)  # (row,col) → (x,y)

    @staticmethod
    def _simple_thin(mask: np.ndarray) -> np.ndarray:
        """简易骨架化 (当 ximgproc 不可用时)"""
        size = np.sum(mask > 0)
        skel = np.zeros(mask.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        img = mask.copy()
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        return skel

    def _compute_confidence(self, length: float, width: float,
                             area: float, aspect_ratio: float) -> float:
        """
        计算划痕检测置信度

        基于多个几何特征的加权评分
        """
        score = 0.0

        # 长度评分: 越长越可信
        if length > 50:
            score += 0.4
        elif length > 30:
            score += 0.2

        # 长宽比评分: 越细长越像划痕
        if aspect_ratio > 5:
            score += 0.3
        elif aspect_ratio > 3:
            score += 0.15

        # 宽度评分: 不宽于阈值
        if width < self.cfg.max_scratch_width_px * 0.5:
            score += 0.2
        elif width < self.cfg.max_scratch_width_px:
            score += 0.1

        # 面积评分
        if area > self.cfg.min_scratch_area_px2 * 2:
            score += 0.1

        return min(score, 1.0)

    # ================================================================
    # 工具方法
    # ================================================================

    def update_calibration(self, camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray):
        """更新标定参数"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def has_calibration(self) -> bool:
        return self.camera_matrix is not None and self.dist_coeffs is not None


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    # 生成模拟划痕图像测试
    test_img = np.ones((600, 800), dtype=np.uint8) * 180  # 灰色金属背景

    # 添加模拟噪声 (金属纹理)
    noise = np.random.normal(0, 8, test_img.shape).astype(np.int16)
    test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 添加几条模拟划痕
    cv2.line(test_img, (100, 150), (400, 160), 40, 2)   # 长划痕 (暗)
    cv2.line(test_img, (200, 300), (350, 310), 30, 1)    # 细划痕
    cv2.line(test_img, (500, 100), (520, 500), 50, 3)   # 垂直划痕
    cv2.line(test_img, (50, 400), (300, 410), 160, 1)   # 亮划痕

    detector = ScratchDetector()
    scratches, mask = detector.detect(test_img)

    print(f"检测到 {len(scratches)} 条划痕:")
    for s in scratches:
        print(f"  #{s.id}: 长度={s.length_px:.1f}px, 宽度={s.width_px:.1f}px, "
              f"角度={s.angle_deg:.1f}°, 面积={s.area_px2:.0f}px², "
              f"长宽比={s.aspect_ratio:.2f}, 置信度={s.confidence:.2f}")

    # 保存结果图像
    result = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    for s in scratches:
        x, y, w, h = s.bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"#{s.id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.drawContours(result, [s.contour for s in scratches], -1, (0, 0, 255), 1)

    from config import OUTPUT_DIR
    out_path = os.path.join(OUTPUT_DIR, "scratch_test_result.png")
    # Use safe_imwrite for Chinese path compatibility
    from config import safe_imwrite
    safe_imwrite(out_path, result)
    print(f"\n测试图像已保存: {out_path}")


