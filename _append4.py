path = r"D:\Pycharm\PyCharm 2024.1.6\jbr\bin\D\PycharmProjects\pythonProject\金属板划痕检测系统\main_app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

extra = '''
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
        for name, var in self.param_vars.items():
            if hasattr(cfg, name):
                setattr(cfg, name, var.get())
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
        self.det_detail.insert(1.0, "\n".join(lines))
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
'''

content += extra
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("done")
