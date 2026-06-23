path = r"D:\Pycharm\PyCharm 2024.1.6\jbr\bin\D\PycharmProjects\pythonProject\金属板划痕检测系统\main_app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

extra = '''
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
            messagebox.showinfo("标定完成",
                "RMS: {} px\n质量: {}\n焦距: fx={}, fy={}\n主点: cx={}, cy={}".format(
                    info["rms_error_px"], info["rms_quality"],
                    info["focal_length"][0], info["focal_length"][1],
                    info["principal_point"][0], info["principal_point"][1]))
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
        self.cal_result.insert(1.0, "\n".join(lines))
        self.cal_result.config(state=tk.DISABLED)
'''

content += extra
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("done")
