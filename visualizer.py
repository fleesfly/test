# -*- coding: utf-8 -*-
"""
可视化模块 - 使用实线标记划痕

视图:
  1. 原图 + 实线标记 (origin)
  2. 处理结果视图 (即实线标记图, 不做三视图)
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from collections import deque

from config import GUI_CFG
from scratch_detector import ScratchInfo


class ScratchVisualizer:
    """
    划痕标记器 - 用实线标记每条划痕的主轴
    """

    def __init__(self, config=None):
        self.cfg = config or GUI_CFG
        self._fps_history = deque(maxlen=30)
        self._proc_time_history = deque(maxlen=30)
        self._last_fps_update = time.time()
        self._fps_counter = 0
        self._current_fps = 0.0

    def annotate(self, image: np.ndarray, scratches: List[ScratchInfo],
                 proc_time_ms: float = 0.0, frame_id: int = 0) -> dict:
        """
        生成两个视图: 原图 + 实线标记结果

        Returns:
            {"origin": 原图, "result": 实线标记结果, "fps": ..., "scratch_count": ...}
        """
        # 确保 3 通道 BGR，否则彩色线条画不上
        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            display = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)
        else:
            display = image.copy()

        # 视图1: 原图 (只加简短状态文字)
        origin_view = self._draw_minimal_status(display.copy(), scratches, frame_id)

        # 视图2: 实线标记结果
        result_view = self._draw_solid_lines(display, scratches)
        result_view = self._draw_status_bar(result_view, scratches, proc_time_ms, frame_id)

        self._update_fps(proc_time_ms)

        total_len = sum(s.length_px for s in scratches) if scratches else 0.0

        return {
            "origin": origin_view,
            "result": result_view,
            "fps": self._current_fps,
            "scratch_count": len(scratches),
            "avg_length": (np.mean([s.length_px for s in scratches])
                           if scratches else 0.0),
            "total_length": total_len,
        }

    # ================================================================
    # 实线标记 (划线痕主轴)
    # ================================================================

    def _draw_solid_lines(self, image: np.ndarray,
                           scratches: List[ScratchInfo]) -> np.ndarray:
        """
        对每条划痕, 用实线标记其主轴方向
        绿色 = 真实划痕, 橙色 = 误检
        """
        vis = image.copy()

        for s in scratches:
            # 真实划痕黄色, 误检蓝色
            color = (0, 255, 255) if not s.false_alarm else (255, 0,0)

            # 使用最小外接矩形的主轴作为实线
            if len(s.contour) >= 5:
                rect = cv2.minAreaRect(s.contour)
                (cx, cy), (w, h), angle = rect

                # 主轴方向: 较长的那条边
                if w >= h:
                    major_len = w
                    rad = np.deg2rad(angle)
                else:
                    major_len = h
                    rad = np.deg2rad(angle + 90)

                # 两端点 (延长 1.2 倍使线更明显)
                half = major_len / 2 * 1.2
                dx = half * np.cos(rad)
                dy = half * np.sin(rad)
                p1 = (int(cx - dx), int(cy - dy))
                p2 = (int(cx + dx), int(cy + dy))

                # 绘制实线 (加粗)
                cv2.line(vis, p1, p2, color, 3, cv2.LINE_AA)
                # 中心圆点
                cv2.circle(vis, (int(cx), int(cy)), 4, color, -1)

                # 标签: 编号 + 长度
                label = f"#{s.id} L={s.length_px:.0f}"
                cv2.putText(vis, label, (int(cx) + 8, int(cy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            else:
                # 轮廓点太少, 直接用centroid画点
                cx, cy = int(s.centroid[0]), int(s.centroid[1])
                cv2.circle(vis, (cx, cy), 4, color, -1)

        return vis

    # ================================================================
    # 状态栏
    # ================================================================

    def _draw_status_bar(self, image: np.ndarray,
                          scratches: List[ScratchInfo],
                          proc_time_ms: float, frame_id: int) -> np.ndarray:
        """检测结果视图的状态栏"""
        vis = image.copy()
        h, w = vis.shape[:2]
        bar_h = 55
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (40, 40, 40), -1)
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

        total_len = sum(s.length_px for s in scratches) if scratches else 0
        avg_conf = (np.mean([s.confidence for s in scratches])
                     if scratches else 0.0)

        lines = [
            f"FPS:{self._current_fps:.1f} | 耗时:{proc_time_ms:.1f}ms | 帧:#{frame_id}",
            f"划痕:{len(scratches)}条 | 总长:{total_len:.1f}px | 平均置信度:{avg_conf:.2f}",
        ]
        for i, line in enumerate(lines):
            y = h - bar_h + 18 + i * 18
            cv2.putText(vis, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

        return vis

    def _draw_minimal_status(self, image: np.ndarray,
                              scratches: List[ScratchInfo],
                              frame_id: int) -> np.ndarray:
        """原图视图的状态栏 (极简)"""
        vis = image.copy()
        h, w = vis.shape[:2]
        info = f"帧:#{frame_id} | 划痕:{len(scratches)}条"
        cv2.putText(vis, info, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return vis

    # ================================================================
    # FPS
    # ================================================================

    def _update_fps(self, proc_time_ms: float):
        self._fps_counter += 1
        self._proc_time_history.append(proc_time_ms)
        now = time.time()
        elapsed = now - self._last_fps_update
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._last_fps_update = now

    @property
    def avg_proc_time(self) -> float:
        return (np.mean(self._proc_time_history)
                if self._proc_time_history else 0.0)

    @property
    def fps(self) -> float:
        return self._current_fps
