path = r"D:\Pycharm\PyCharm 2024.1.6\jbr\bin\D\PycharmProjects\pythonProject\金属板划痕检测系统\main_app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

extra = '''
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
                self.root.after(60, self._update_frame)
                return
            frame = self._latest_frame.copy()

        # 当前标签页
        cur = self.notebook.index(self.notebook.select())

        if cur == 0:
            # 标定页: 显示实时画面
            self._show_on_canvas(self.cal_canvas, frame, "_cal_tk")
        else:
            # 检测页: 显示原图, 如果检测中则显示结果
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
            return
        h, w = img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            return
        resized = cv2.resize(img, (nw, nh))
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
'''

content += extra
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("done")
