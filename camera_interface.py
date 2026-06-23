# -*- coding: utf-8 -*-
"""
Allied Vision 相机接口模块 (基于 Vimba SDK / VmbPy)
"""

import os
import time
import threading
import atexit
from queue import Queue, Empty
from typing import Optional, Callable, Any
import numpy as np

try:
    from vmbpy import *
    VMBPY_AVAILABLE = True
except ImportError:
    VMBPY_AVAILABLE = False
    print("[警告] vmbpy 未安装。请运行: pip install vmbpy")

from config import CAM_CFG


class CameraInterface:

    def __init__(self, camera_id: Optional[str] = None, config=None):
        self.cfg = config or CAM_CFG
        self.camera_id = camera_id
        self._vmb: Optional[Any] = None
        self._cam: Optional[Any] = None
        self._streaming = False
        self._frame_queue = Queue(maxsize=16)
        self._frame_callback: Optional[Callable] = None
        self._error_count = 0
        self._max_errors = 10
        self._trigger_thread: Optional[threading.Thread] = None
        self._trigger_interval = 0.1
        self._saved_settings: dict = {}
        self._streaming_was_active = False
        atexit.register(self._cleanup_vmb)

    def __del__(self):
        self._cleanup_vmb()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> bool:
        if not VMBPY_AVAILABLE:
            print("[相机] 错误: vmbpy未安装")
            return True

        genTL = os.environ.get("GENICAM_GENTL64_PATH", "")
        if "VimbaX" in genTL and "Vimba 6.0" in genTL:
            print("[相机] 警告: GENICAM_GENTL64_PATH 同时包含 VimbaX 和 Vimba 6.0 路径，可能冲突")

        for attempt in range(1, 4):
            try:
                self._vmb = VmbSystem.get_instance()
                self._vmb.__enter__()
                cameras = self._vmb.get_all_cameras()

                if not cameras:
                    print(f"[相机] 未检测到相机 (尝试 {attempt}/3)")
                    self._vmb.__exit__(None, None, None)
                    self._vmb = None
                    if attempt < 3:
                        time.sleep(3)
                        continue
                    print("[相机] 请检查: ①USB线是否连接 ②Vimba驱动是否安装 ③是否有其他程序占用")
                    return False

                self._cam = cameras[0] if not self.camera_id else \
                    next((c for c in cameras if c.get_id() == self.camera_id), None)
                if self._cam is None:
                    print(f"[相机] 未找到指定相机: {self.camera_id}")
                    return False

                permitted = self._cam.get_permitted_access_modes()
                if AccessMode.Full in permitted:
                    self._cam.__enter__()
                    print(f"[相机] [OK] 已连接: {self._cam.get_name()}")
                    self._configure_camera()
                    return True
                elif AccessMode.Read in permitted:
                    print(f"[相机] 相机只能以只读模式访问，尝试打开...")
                    self._cam.__enter__()
                    print(f"[相机] [OK] 已连接 (只读): {self._cam.get_name()}")
                    self._configure_camera()
                    return True
                else:
                    print(f"[相机] [X] 相机被占用 (尝试 {attempt}/3)")
                    print(f"[相机]   允许模式: {permitted}")
                    try:
                        self._vmb.__exit__(None, None, None)
                    except:
                        pass
                    self._vmb = None
                    self._cam = None
                    if attempt < 3:
                        time.sleep(3)
                        continue
                    return False

            except VmbCameraError as e:
                print(f"[相机] [X] 连接失败 (尝试 {attempt}/3): {e}")
                self._cleanup_vmb()
                if attempt < 3:
                    time.sleep(3)
            except Exception as e:
                print(f"[相机] [X] 未知错误: {e}")
                self._cleanup_vmb()
                return False

        print("[相机] 连接失败，请检查 Vimba SDK 和相机连接")
        return False

    def _cleanup_vmb(self):
        if self._streaming:
            self.stop_streaming()
        if self._cam:
            try:
                self._cam.__exit__(None, None, None)
            except:
                pass
            self._cam = None
        if self._vmb:
            try:
                self._vmb.__exit__(None, None, None)
            except:
                pass
            self._vmb = None

    def close(self):
        self._cleanup_vmb()
        try:
            atexit.unregister(self._cleanup_vmb)
        except:
            pass

    def _configure_camera(self):
        """配置相机: 先关闭自动功能, 设置 Freerun 连续采集模式以匹配 Vimba Viewer"""
        try:
            # 0. 关闭所有自动功能 (Vimba Viewer 默认也是 Off)
            for auto_feat in ["ExposureAuto", "GainAuto", "BalanceWhiteAuto"]:
                try:
                    feat = self._cam.get_feature_by_name(auto_feat)
                    if feat is not None:
                        feat.set("Off")
                except Exception:
                    pass

            # 1. 先设分辨率（必须在触发模式之前）
            if self.cfg.width > 0 and self.cfg.height > 0:
                self._set_feature("Width", self.cfg.width)
                self._set_feature("Height", self.cfg.height)
            if self.cfg.offset_x > 0:
                self._set_feature("OffsetX", self.cfg.offset_x)
            if self.cfg.offset_y > 0:
                self._set_feature("OffsetY", self.cfg.offset_y)

            # 2. 设为 Freerun (Continuous) 触发模式 - 与 Vimba Viewer 默认行为一致
            try:
                self._set_feature("TriggerMode", "Off")
                self._set_feature("AcquisitionMode", "Continuous")
            except Exception:
                pass

            # 3. 像素格式固定为 Mono8
            self._set_feature("PixelFormat", "BayerRG8")

        except Exception as e:
            print(f"[相机] 参数配置警告: {e}")

    def save_settings(self):
        """保存当前相机关键设置, 以便后续恢复"""
        if self._cam is None:
            return
        for name in ["ExposureTime", "Gain", "PixelFormat",
                      "Width", "Height", "OffsetX", "OffsetY"]:
            try:
                feat = self._cam.get_feature_by_name(name)
                if feat is not None:
                    self._saved_settings[name] = feat.get()
            except Exception:
                pass

    def restore_settings(self):
        """恢复相机设置为保存时的值"""
        if self._cam is None or not self._saved_settings:
            return
        was_streaming = self._streaming
        if was_streaming:
            self.stop_streaming()
            time.sleep(0.05)
        for name, val in self._saved_settings.items():
            self._set_feature(name, val)
        if was_streaming and self._frame_callback:
            self.start_streaming(self._frame_callback, 30.0)

    def load_xml_settings(self, filepath: str) -> bool:
        """从 Vimba Viewer 导出的 XML 文件加载完整相机配置"""
        if self._cam is None:
            print("[相机] 错误: 相机未连接，无法加载 XML 设置")
            return False
        try:
            # 需要确保 vmbpy 的 PersistenceType 已导入
            from vmbpy import PersistType
            was_streaming = self._streaming
            if was_streaming:
                self.stop_streaming()
                time.sleep(0.05)
            self._cam.load_settings(filepath, PersistType.All)
            print(f"[相机] 已加载 XML 设置: {filepath}")
            # 同步 cfg 中的曝光/增益值
            for name in ["ExposureTime", "Gain"]:
                try:
                    feat = self._cam.get_feature_by_name(name)
                    if feat is not None:
                        val = feat.get()
                        if name == "ExposureTime":
                            self.cfg.exposure_time_us = float(val)
                        elif name == "Gain":
                            self.cfg.gain_db = float(val)
                except:
                    pass
            if was_streaming and self._frame_callback:
                self.start_streaming(self._frame_callback, 30.0)
            return True
        except Exception as e:
            print(f"[相机] 加载 XML 设置失败: {e}")
            return False

    def _set_feature(self, feature_name: str, value):
        if self._cam is None:
            return
        try:
            feat = self._cam.get_feature_by_name(feature_name)
            if feat is not None:
                feat.set(value)
        except Exception:
            pass

    def set_exposure(self, exposure_us: float):
        """设置曝光值（自动 stop/restart 流，防止 mid-frame 修改导致条带）"""
        print(f"[DEBUG Exposure] 请求值: {exposure_us}µs | 当前cfg: {self.cfg.exposure_time_us}µs")
        was_streaming = self._streaming
        if was_streaming:
            self.stop_streaming()
            time.sleep(0.03)
        try:
            feat = self._cam.get_feature_by_name("ExposureTime")
            if feat is not None:
                old = feat.get()
                feat.set(exposure_us)
                new = feat.get()
                print(f"[DEBUG Exposure] 硬件 {old}µs → {new}µs (请求 {exposure_us}µs)")
            self.cfg.exposure_time_us = exposure_us
        except Exception as e:
            print(f"[DEBUG Exposure] 设置失败: {e}")
        if was_streaming and self._frame_callback:
            self.start_streaming(self._frame_callback, 30.0)

    def set_gain(self, gain_db: float):
        """设置增益值（自动 stop/restart 流，防止 mid-frame 修改导致条带）"""
        print(f"[DEBUG Gain] 请求值: {gain_db}dB | 当前cfg: {self.cfg.gain_db}dB")
        was_streaming = self._streaming
        if was_streaming:
            self.stop_streaming()
            time.sleep(0.03)
        try:
            feat = self._cam.get_feature_by_name("Gain")
            if feat is not None:
                old = feat.get()
                feat.set(gain_db)
                new = feat.get()
                print(f"[DEBUG Gain] 硬件 {old}dB → {new}dB (请求 {gain_db}dB)")
            self.cfg.gain_db = gain_db
        except Exception as e:
            print(f"[DEBUG Gain] 设置失败: {e}")
        if was_streaming and self._frame_callback:
            self.start_streaming(self._frame_callback, 30.0)

    def _trigger_loop(self):
        while self._streaming and self._cam is not None:
            try:
                self._cam.TriggerSoftware.run()
            except Exception:
                pass
            time.sleep(self._trigger_interval)

    def start_streaming(self, frame_callback: Callable[[np.ndarray], None],
                        trigger_hz: float = 30.0):
        """启动连续采集 (Freerun), 匹配 Vimba Viewer 行为"""
        self._frame_callback = frame_callback

        if self._cam is None:
            print("[相机] 错误: 相机未连接")
            return

        try:
            # 确保 Freerun 模式
            try:
                self._set_feature("TriggerMode", "Off")
                self._set_feature("AcquisitionMode", "Continuous")
            except Exception:
                pass
            self._cam.start_streaming(self._frame_handler, buffer_count=10)
            self._streaming = True
            print(f"[相机] 采集已启动 (连续采集 Freerun)")
        except Exception as e:
            print(f"[相机] 启动采集失败: {e}")
            self._streaming = False

    def stop_streaming(self):
        self._streaming = False
        if self._trigger_thread and self._trigger_thread.is_alive():
            self._trigger_thread.join(timeout=1.0)
            self._trigger_thread = None
        if self._cam:
            try:
                self._cam.stop_streaming()
            except Exception:
                pass

    def _frame_handler(self, cam, stream, frame):
        if frame.get_status() == FrameStatus.Complete:
            try:
                img = frame.as_numpy_ndarray()
                img_copy = img.copy()

                if not self._frame_queue.full():
                    self._frame_queue.put(img_copy)
                else:
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass
                    self._frame_queue.put(img_copy)

                if self._frame_callback:
                    try:
                        self._frame_callback(img_copy)
                    except Exception:
                        pass

                self._error_count = 0
            except Exception:
                self._error_count += 1
                if self._error_count > self._max_errors:
                    self.stop_streaming()

        cam.queue_frame(frame)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def is_connected(self) -> bool:
        return self._cam is not None
