# -*- coding: utf-8 -*-
"""
保存当前相机硬件参数到 XML 文件
用法：关掉 Vimba Viewer 后运行本脚本
      python save_cam_settings.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vmbpy import *
except ImportError:
    print("[错误] 未安装 vmbpy，请运行: pip install vmbpy")
    sys.exit(1)

def main():
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_settings.xml")
    print("=" * 50)
    print("  保存相机参数到 XML")
    print("=" * 50)
    print(f"  输出: {save_path}")
    print()

    vmb = VmbSystem.get_instance()
    vmb.__enter__()
    try:
        cameras = vmb.get_all_cameras()
        if not cameras:
            print("[错误] 未检测到相机")
            return False

        cam = cameras[0]
        print(f"[相机] {cam.get_name()}")

        permitted = cam.get_permitted_access_modes()
        if AccessMode.Full not in permitted and AccessMode.Read not in permitted:
            print("[错误] 相机被占用（请关闭 Vimba Viewer 等程序）")
            return False

        cam.__enter__()
        try:
            # 停止可能正在运行的采流
            try:
                cam.stop_streaming()
            except:
                pass
            time.sleep(0.1)

            # 保存完整参数到 XML
            cam.save_settings(save_path, PersistType.All)
            print(f"[OK] 已保存: {save_path}")
            print(f"     文件大小: {os.path.getsize(save_path)} bytes")
            return True

        finally:
            cam.__exit__(None, None, None)
    finally:
        vmb.__exit__(None, None, None)


if __name__ == "__main__":
    success = main()
    print()
    if success:
        print("完成！现在可以在 main_app.py 中使用以下代码加载：")
        print('  self.camera.load_xml_settings("camera_settings.xml")')
    else:
        print("保存失败")
    input("按 Enter 退出...")
