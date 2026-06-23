import os, sys, time, json, threading
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

from config import (
    CALIB_CFG, CAM_CFG, SCRATCH_CFG, GUI_CFG,
    CALIB_IMAGES_DIR, OUTPUT_DIR, CALIB_PARAMS_FILE,
    safe_imwrite
)
from camera_calibration import HalconDotCalibrator
from scratch_detector import ScratchDetector
from camera_interface import CameraInterface
from visualizer import ScratchVisualizer


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(GUI_CFG.window_title)
        self.root.geometry(f"{GUI_CFG.window_width}x{GUI_CFG.window_height}")
        self.root.minsize(1400, 850)

        self.calibrator = HalconDotCalibrator()
        self.detector = ScratchDetector()
        self.vis = ScratchVisualizer()
        self.camera: CameraInterface = None

        self.streaming = False
        self.detecting = False
        self.calibration_loaded = False
        self.frame_counter = 0

        self._latest_frame = None
        self._frame_lock = threading.Lock()

        self.calib_images = []

        self._build_ui()
        self._try_load_calibration()
        print("[系统] 初始化完成")

    # ================================================================
    # UI 构建
    # ================================================================

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        self._build_calibration_tab()
        self._build_detection_tab()
        self._build_status_bar()

    # ================================================================
    # 标签页1: 相机标定
    # ================================================================

    def _build_calibration_tab(self):
        tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab, text="📷 ????")

        # ===== ????? (Vimba Viewer ??: ??) =====
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(toolbar, text="🔌 ????", command=self._connect_camera, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="▶ ????", command=self._start_streaming, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="📷 ????", command=self._capture_calib, width=10).pack(side=tk.LEFT, padx=1)
        self.cal_cnt = ttk.Label(toolbar, text="0/10", foreground="#aaa", font=("", 9))
        self.cal_cnt.pack(side=tk.LEFT, padx=6)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(toolbar, text="??:", font=("", 8)).pack(side=tk.LEFT)
        self.exposure_var = tk.IntVar(value=int(CAM_CFG.exposure_time_us))
        self.exposure_scale = ttk.Scale(toolbar, from_=100, to=100000, orient=tk.HORIZONTAL,
                                         variable=self.exposure_var, length=140)
        self.exposure_scale.pack(side=tk.LEFT, padx=2)
        self.exposure_scale.bind('<ButtonRelease-1>', lambda e: self._apply_cam_params())
        self.exposure_label = ttk.Label(toolbar, text=str(self.exposure_var.get()), width=6, font=("", 8))
        self.exposure_label.pack(side=tk.LEFT)
        ttk.Label(toolbar, text="??:", font=("", 8)).pack(side=tk.LEFT, padx=(6, 0))
        self.gain_var = tk.DoubleVar(value=CAM_CFG.gain_db)
        self.gain_scale = ttk.Scale(toolbar, from_=0, to=24, orient=tk.HORIZONTAL,
                                     variable=self.gain_var, length=100)
        self.gain_scale.pack(side=tk.LEFT, padx=2)
        self.gain_scale.bind('<ButtonRelease-1>', lambda e: self._apply_cam_params())
        self.gain_label = ttk.Label(toolbar, text=f"{self.gain_var.get():.1f}", width=4, font=("", 8))
        self.gain_label.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="??", command=self._apply_cam_params, width=5).pack(side=tk.LEFT, padx=4)
        self.exposure_var.trace_add("write", lambda *_: self.exposure_label.config(text=str(self.exposure_var.get())))
        self.gain_var.trace_add("write", lambda *_: self.gain_label.config(text=f"{self.gain_var.get():.1f}"))

        # ===== ???: ????? + ???? =====
        main = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # --- ??: ???? (?, ??) ---
        left_frame = ttk.Frame(main)
        main.add(left_frame, weight=3)
        self.cal_canvas = tk.Canvas(left_frame, bg="#0d0d0d",
                                     highlightthickness=1, highlightbackground="#444")
        self.cal_canvas.pack(fill=tk.BOTH, expand=True)
        self._cal_tk = None

        # --- ??: ???? (280px ????) ---
        right_frame = ttk.Frame(main, width=280)
        main.add(right_frame, weight=0)
        right_frame.pack_propagate(False)

        ttk.Label(right_frame, text="📋 ????", font=("", 9, "bold")).pack(anchor="w", pady=(0, 2))
        lst_f = ttk.Frame(right_frame)
        lst_f.pack(fill=tk.BOTH, expand=True)
        self.cal_list = tk.Listbox(lst_f, height=5, bg="#1e1e1e", fg="#d4d4d4")
        sc = ttk.Scrollbar(lst_f, orient=tk.VERTICAL, command=self.cal_list.yview)
        self.cal_list.configure(yscrollcommand=sc.set)
        self.cal_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sc.pack(side=tk.RIGHT, fill=tk.Y)

        self.cal_progress = ttk.Progressbar(right_frame, mode="determinate", maximum=CALIB_CFG.min_calib_images)
        self.cal_progress.pack(fill=tk.X, pady=2)

        cal_btns = ttk.Frame(right_frame)
        cal_btns.pack(fill=tk.X, pady=2)
        ttk.Button(cal_btns, text="????", command=self._run_calibration, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(cal_btns, text="????", command=self._save_calibration, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(cal_btns, text="????", command=self._load_calibration, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(cal_btns, text="??", command=self._reset_calibration, width=8).pack(side=tk.LEFT, padx=1)

        ttk.Label(right_frame, text="????", font=("", 9, "bold")).pack(anchor="w", pady=(4, 2))
        self.cal_result = tk.Text(right_frame, wrap=tk.NONE, bg="#1e1e1e", fg="#d4d4d4",
                                   font=("Consolas", 9), height=8)
        self.cal_result.pack(fill=tk.BOTH, expand=True)
        self.cal_result.insert(1.0, "??????...\n")
        self.cal_result.config(state=tk.DISABLED)

    # ================================================================
    # 标签页2: 划痕检测
    # ================================================================

    def _build_detection_tab(self):
        tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab, text="🔍 ????")

        # ===== ????? (??, ??) =====
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        self.det_start_btn = ttk.Button(toolbar, text="▶ ????", command=self._start_detection, width=10)
        self.det_start_btn.pack(side=tk.LEFT, padx=1)
        self.det_stop_btn = ttk.Button(toolbar, text="⏹ ????", command=self._stop_detection, width=10, state=tk.DISABLED)
        self.det_stop_btn.pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="📸 ??", command=self._screenshot, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # ????
        self.param_vars = {}
        params = [
            ("tophat_kernel_size", "TopHat", 3, 99, int(SCRATCH_CFG.tophat_kernel_size)),
            ("canny_low", "Canny?", 1, 255, SCRATCH_CFG.canny_low),
            ("canny_high", "Canny?", 1, 255, SCRATCH_CFG.canny_high),
            ("min_scratch_length_px", "????", 5, 500, int(SCRATCH_CFG.min_scratch_length_px)),
            ("gabor_ksize", "Gabor?", 3, 51, int(SCRATCH_CFG.gabor_ksize)),
        ]
        for name, label, lo, hi, default in params:
            f = ttk.Frame(toolbar)
            f.pack(side=tk.LEFT, padx=2)
            ttk.Label(f, text=label, font=("", 7)).pack()
            var = tk.DoubleVar(value=default)
            self.param_vars[name] = var
            length = 60 if (isinstance(default, float) and (hi - lo) < 10) else 80
            s = ttk.Scale(f, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var, length=length)
            s.pack()
        ttk.Button(toolbar, text="????", command=self._apply_params, width=8).pack(side=tk.LEFT, padx=4)

        # ===== ???: ????? =====
        main = ttk.Frame(tab)
        main.pack(fill=tk.BOTH, expand=True)

        # --- ??: ???? ---
        lf = ttk.Frame(main)
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        ttk.Label(lf, text="📹 ???? (???)", font=("", 9, "bold")).pack(anchor="w")
        self.det_orig = tk.Canvas(lf, bg="#0d0d0d",
                                   highlightthickness=1, highlightbackground="#444")
        self.det_orig.pack(fill=tk.BOTH, expand=True)
        self._det_orig_tk = None

        # --- ??: ???? ---
        rf = ttk.Frame(main)
        rf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        ttk.Label(rf, text="🔍 ??????", font=("", 9, "bold")).pack(anchor="w")
        self.det_result = tk.Canvas(rf, bg="#0d0d0d",
                                    highlightthickness=1, highlightbackground="#444")
        self.det_result.pack(fill=tk.BOTH, expand=True)
        self._det_result_tk = None

        # ===== ????? =====
        self.det_detail = tk.Text(tab, height=4, wrap=tk.NONE, bg="#1e1e1e", fg="#d4d4d4",
                                   font=("Consolas", 9))
        self.det_detail.pack(fill=tk.X, pady=(4, 0))
        self.det_detail.insert(1.0, "??????...")
        self.det_detail.config(state=tk.DISABLED)

    def _build_status_bar(self):
        bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.cam_info_var = tk.StringVar(value="相机: 未连接")
        ttk.Label(bar, textvariable=self.cam_info_var, font=("", 8)).pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.status_var, font=("", 8)).pack(side=tk.RIGHT, padx=5)

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ================================================================
    # 相机操作
    # ================================================================

    def _connect_camera(self):
        self._set_status("连接相机...")
        try:
            self.camera = CameraInterface()
            if self.camera.open():
                ci = "已加载" if self.calibration_loaded else "未加载"
                self.cam_info_var.set("相机: 已连接 | 标定: " + ci)
                self._set_status("相机已连接")
                self.exposure_var.set(int(self.camera.cfg.exposure_time_us))
                self.gain_var.set(self.camera.cfg.gain_db)
            else:
                self._set_status("连接失败")
                self.camera = None
                self.cam_info_var.set("相机: 连接失败")
        except Exception as e:
            self._set_status("连接错误: " + str(e))
            self.camera = None
            self.cam_info_var.set("相机: 连接失败")

    def _apply_cam_params(self):
        if self.camera is None:
            messagebox.showwarning("提示", "请先连接相机")
            return
        exposure = self.exposure_var.get()
        gain = self.gain_var.get()
        try:
            self.camera.set_exposure(exposure)
            self.camera.set_gain(gain)
            self._set_status(f"参数已应用: 曝光={exposure}us 增益={gain:.1f}dB")
        except Exception as e:
            self._set_status(f"参数应用失败: {e}")

    def _start_streaming(self):
        if self.streaming:
            return
        if self.camera is None:
            self._connect_camera()
        self.streaming = True
        if self.camera:
            self.camera.start_streaming(self._on_frame, trigger_hz=10.0)
        self._set_status("预览中")
        self._frame_check_count = 0
        self._update_frame()

    def _stop_streaming(self):
        self.streaming = False
        self.detecting = False
        if self.camera:
            self.camera.stop_streaming()

    # ================================================================
    # 帧处理
    # ================================================================

    def _on_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame.copy()
        self.frame_counter += 1

    def _update_frame(self):
        if not self.streaming:
            return
        with self._frame_lock:
            if self._latest_frame is None:
                self._frame_check_count += 1
                if self._frame_check_count > 50:
                    self._set_status("等待相机帧...")
                    self.root.after(60, self._update_frame)
                    return
                else:
                    self.root.after(60, self._update_frame)
                    return
            frame = self._latest_frame.copy()

        cur = self.notebook.index(self.notebook.select())

        if cur == 0:
            self._show_on_canvas(self.cal_canvas, frame, "_cal_tk")
        else:
            self._show_on_canvas(self.det_orig, frame, "_det_orig_tk")
            if self.detecting:
                t0 = time.time()
                scratches, mask = self.detector.detect(frame)
                dt = (time.time() - t0) * 1000
                views = self.vis.annotate(frame, scratches, dt, self.frame_counter)
                self._show_on_canvas(self.det_result, views["result"], "_det_result_tk")
                self._update_detail(scratches, views)
            else:
                self._show_on_canvas(self.det_result, frame, "_det_result_tk")

        self.root.after(60, self._update_frame)

    def _show_on_canvas(self, canvas, img, ref):
        if img is None:
            return
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 2 or ch <= 2:
            self.root.after(100, lambda: self._show_on_canvas(canvas, img, ref))
            return
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            return
        interp = cv2.INTER_LANCZOS4 if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        if len(resized.shape) == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        canvas.create_image(x0, y0, anchor=tk.NW, image=tk_img)
        setattr(self, ref, tk_img)

    # ================================================================
    # 标定操作
    # ================================================================

    def _capture_calib(self):
        with self._frame_lock:
            if self._latest_frame is None:
                messagebox.showwarning("提示", "请先点击「开始预览」")
                return
            frame = self._latest_frame.copy()
        ok, centers, annotated = self.calibrator.detect_dots(frame)
        if ok:
            self.calibrator.add_calibration_image(frame, centers, annotated)
            n = self.calibrator.num_captured
            self.cal_list.insert(tk.END, "图像 {:02d} - {}个点".format(n, len(centers)))
            self.cal_list.see(tk.END)
            self.cal_progress["value"] = n
            self.cal_cnt.config(text="{}/{}".format(n, CALIB_CFG.min_calib_images))
            self._set_status("标定图像 {} 已采集".format(n))
        else:
            self._set_status("检测失败 - 标定板需完整可见, 调整光源")

    def _run_calibration(self):
        n = self.calibrator.num_captured
        if n < CALIB_CFG.min_calib_images:
            messagebox.showwarning("提示", "至少需要 {} 张, 当前 {}".format(CALIB_CFG.min_calib_images, n))
            return
        self._set_status("标定中...")
        self.root.update()
        ok = self.calibrator.calibrate()
        if ok:
            info = self.calibrator.evaluate_calibration()
            self.detector.update_calibration(self.calibrator.camera_matrix,
                                              self.calibrator.dist_coeffs)
            self.calibration_loaded = True
            self.cam_info_var.set("相机: 已连接 | 标定: 已加载")
            self._show_calib_result(info)
            msg = "RMS: {} px\n质量: {}\n焦距: fx={}, fy={}\n主点: cx={}, cy={}".format(
                info["rms_error_px"], info["rms_quality"],
                info["focal_length"][0], info["focal_length"][1],
                info["principal_point"][0], info["principal_point"][1])
            messagebox.showinfo("标定完成", msg)
        else:
            messagebox.showerror("标定失败", "请检查标定图像质量")

    def _show_calib_result(self, info):
        lines = []
        lines.append("=" * 50)
        lines.append("  标定结果 (张正友标定法)")
        lines.append("=" * 50)
        lines.append("  RMS: {} px  [{}]".format(info["rms_error_px"], info["rms_quality"]))
        lines.append("")
        lines.append("  [内参]")
        lines.append("  焦距: fx={:.1f}, fy={:.1f}".format(info["focal_length"][0], info["focal_length"][1]))
        lines.append("  主点: cx={:.1f}, cy={:.1f}".format(info["principal_point"][0], info["principal_point"][1]))
        d = info["distortion_coefficients"]
        lines.append("  畸变: k1={}, k2={}".format(d["k1 (径向)"], d["k2 (径向)"]))
        lines.append("         p1={}, p2={}".format(d["p1 (切向)"], d["p2 (切向)"]))
        lines.append("")
        lines.append("  相机矩阵:")
        cm = info["camera_matrix"]
        lines.append("    [{:.1f}, {:.1f}, {:.1f}]".format(cm[0][0], cm[0][1], cm[0][2]))
        lines.append("    [   0   , {:.1f}, {:.1f}]".format(cm[1][1], cm[1][2]))
        lines.append("    [   0   ,    0   ,    1   ]")
        lines.append("")
        lines.append("  [外参 - 每张图像]")
        lines.append("  图像 | 平移(mm) | 误差(px)")
        lines.append("  " + "-" * 30)
        for ext in info["extrinsics_summary"]:
            lines.append("  #{:02d}  | {:9.1f}  | {:.4f}".format(
                ext["image_index"], ext["translation_mm"], ext["reprojection_error_px"]))
        lines.append("")
        lines.append("  最佳: #{} | 最差: #{}".format(info["best_image_index"], info["worst_image_index"]))
        lines.append("")
        lines.append("  [逐步操作提示]")
        lines.append("  1. 切换到「划痕检测」标签页")
        lines.append("  2. 点击「开始检测」运行实时检测")
        lines.append("  3. 调节算法参数优化检测效果")

        self.cal_result.config(state=tk.NORMAL)
        self.cal_result.delete(1.0, tk.END)
        self.cal_result.insert(1.0, "等待执行标定...\n")
        self.cal_result.config(state=tk.DISABLED)

    # ================================================================
    # 检测操作
    # ================================================================

    def _start_detection(self):
        if not self.streaming:
            self._start_streaming()
            self.root.after(200, self._start_detection)
            return
        self.detecting = True
        self.det_start_btn.config(state=tk.DISABLED)
        self.det_stop_btn.config(state=tk.NORMAL)
        self.notebook.select(1)
        self._set_status("划痕检测运行中")

    def _stop_detection(self):
        self.detecting = False
        self.det_start_btn.config(state=tk.NORMAL)
        self.det_stop_btn.config(state=tk.DISABLED)
        self._set_status("检测已停止")

    def _apply_params(self):
        cfg = SCRATCH_CFG
        hints = type(cfg).__annotations__
        for name, var in self.param_vars.items():
            if hasattr(cfg, name):
                val = var.get()
                if name in hints and hints[name] == int:
                    val = int(val)
                setattr(cfg, name, val)
        self.detector.cfg = cfg
        if cfg.enable_clahe:
            self.detector._clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip_limit,
                tileGridSize=(cfg.clahe_tile_size, cfg.clahe_tile_size))
        self.detector._init_gabor_kernels()
        self._set_status("参数已更新")

    def _update_detail(self, scratches, views):
        lines = ["FPS: {:.1f} | {}条划痕 | 总长: {:.1f}px".format(
            views["fps"], views["scratch_count"], views["total_length"])]
        if scratches:
            hdr = "{:<4} {:>8} {:>8} {:>7} {:>7}".format("ID", "长度px", "宽度px", "角度", "置信度")
            lines.append(hdr)
            for s in scratches[:8]:
                lines.append("#{:<3} {:>8.1f} {:>8.1f} {:>7.1f} {:>7.2f}".format(
                    s.id, s.length_px, s.width_px, s.angle_deg, s.confidence))
        self.det_detail.config(state=tk.NORMAL)
        self.det_detail.delete(1.0, tk.END)
        self.det_detail.insert(1.0, "".join(lines))
        self.det_detail.config(state=tk.DISABLED)

    def _screenshot(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, "scratch_{}.png".format(ts))
        safe_imwrite(path, frame)
        self._set_status("截图: " + path)

    # ================================================================
    # 工具
    # ================================================================

    def _save_calibration(self):
        if self.calibrator.camera_matrix is None:
            messagebox.showwarning("提示", "尚未执行标定")
            return
        fp = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(CALIB_PARAMS_FILE),
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if fp:
            self.calibrator.save(fp)
            self._set_status("已保存: " + fp)

    def _load_calibration(self):
        ok = self.calibrator.load()
        if ok:
            self.detector.update_calibration(self.calibrator.camera_matrix,
                                              self.calibrator.dist_coeffs)
            self.calibration_loaded = True
            info = self.calibrator.evaluate_calibration()
            self.cam_info_var.set("相机: 未连接 | 标定: 已加载")
            self._show_calib_result(info)
            messagebox.showinfo("加载成功", "RMS误差: {} px".format(info["rms_error_px"]))
        else:
            messagebox.showwarning("提示", "未找到标定参数文件")

    def _try_load_calibration(self):
        if self.calibrator.load():
            self.detector.update_calibration(self.calibrator.camera_matrix,
                                              self.calibrator.dist_coeffs)
            self.calibration_loaded = True
            info = self.calibrator.evaluate_calibration()
            self.cam_info_var.set("相机: 未连接 | 标定: 已加载")
            self._show_calib_result(info)
            self._set_status("已自动加载标定 (RMS: {} px)".format(info["rms_error_px"]))

    def _reset_calibration(self):
        self.calibrator.reset()
        self.cal_list.delete(0, tk.END)
        self.cal_progress["value"] = 0
        self.cal_cnt.config(text="0/10")
        self.calibration_loaded = False
        self.cam_info_var.set("相机: 已连接 | 标定: 未加载")
        self._set_status("标定已重置")
        self.cal_result.config(state=tk.NORMAL)
        self.cal_result.delete(1.0, tk.END)
        self.cal_result.insert(tk.END, "采集 10 张以上标定图像后点击「执行标定」\n")
        self.cal_result.config(state=tk.DISABLED)

    def on_closing(self):
        self._stop_streaming()
        if self.camera:
            self.camera.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

