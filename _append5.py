path = r"D:\Pycharm\PyCharm 2024.1.6\jbr\bin\D\PycharmProjects\pythonProject\金属板划痕检测系统\main_app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

extra = '''
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

    def _set_status(self, msg):
        self.status_var.set(msg)

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
'''

content += extra
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("done")
