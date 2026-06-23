path = r"D:\Pycharm\PyCharm 2024.1.6\jbr\bin\D\PycharmProjects\pythonProject\金属板划痕检测系统\main_app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

extra = '''
    def _connect_camera(self):
        self._set_status("连接相机...")
        try:
            self.camera = CameraInterface()
            if self.camera.open():
                ci = "已加载" if self.calibration_loaded else "未加载"
                self.cam_info_var.set("相机: 已连接 | 标定: " + ci)
                self._set_status("相机已连接")
            else:
                self._set_status("连接失败, 使用模拟模式")
                self.camera = CameraInterface()
                self.cam_info_var.set("相机: 模拟模式")
        except Exception as e:
            self._set_status("连接错误: " + str(e))
            self.camera = CameraInterface()
            self.cam_info_var.set("相机: 模拟模式")

    def _start_streaming(self):
        if self.streaming:
            return
        if self.camera is None:
            self._connect_camera()
        self.streaming = True
        self.camera.start_streaming(self._on_frame)
        self._set_status("预览中")
        self._update_frame()

    def _stop_streaming(self):
        self.streaming = False
        self.detecting = False
        if self.camera:
            self.camera.stop_streaming()
'''

content += extra
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("done")
