# -*- coding: utf-8 -*-
import time
import os
import sys
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CALIB_CFG, CALIB_IMAGES_DIR, OUTPUT_DIR, CALIB_PARAMS_FILE, safe_imwrite
from camera_calibration import HalconDotCalibrator
from camera_interface import CameraInterface

cal = HalconDotCalibrator()
cam = CameraInterface()
print("[1] Connecting camera...")
if not cam.open():
    print("FAILED: Cannot connect camera")
    exit(1)

cam.start_streaming(None)
print("    Warming up...", end="", flush=True)
frame = None
for _ in range(20):
    frame = cam.get_frame(timeout=2.0)
    if frame is not None:
        break
    time.sleep(0.1)
if frame is None:
    print(" FAILED")
    cam.close()
    exit(1)
print(" OK")
h, w = frame.shape[:2]
print(f"    Image: {w}x{h}")

captured = 0
show_ov = True
running = True
last_msg = ""
last_t = 0

cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration Capture", 960, 720)
print("Controls: [Space]Capture [c]Calibrate [r]Reset [q]Quit")
while running:
    frame = cam.get_frame(timeout=0.5)
    if frame is None:
        continue
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()
    else:
        gray = frame.copy()
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    col = (0, 255, 0) if captured >= 10 else (0, 255, 255)
    t1 = f"Captured: {captured}/10"
    cv2.putText(disp, t1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    if show_ov:
        ok, ctrs, ann = cal.detect_dots(gray)
        if ok:
            disp = ann
            cv2.putText(disp, "DETECTED", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "NOT DETECTED", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if last_msg and time.time() - last_t < 2.5:
        cv2.putText(disp, last_msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    help_text = "[Space]Capture [c]Calibrate [r]Reset [q]Quit"
    cv2.putText(disp, help_text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("Calibration Capture", disp)
    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord("q"):
        running = False
    elif key == ord(" ") or key == 13:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        idx = captured + 1
        ok, ctrs, ann = cal.detect_dots(gray)
        if not ok:
            last_msg = "No dots detected"
            last_t = time.time()
            print(f"  [{idx}] FAIL - no dots")
            continue
        cal.add_calibration_image(gray, ctrs, ann)
        captured += 1
        raw_p = os.path.join(CALIB_IMAGES_DIR, f"calib_{idx:02d}_{ts}_original.png")
        det_p = os.path.join(CALIB_IMAGES_DIR, f"calib_{idx:02d}_{ts}_detected.png")
        safe_imwrite(raw_p, gray)
        safe_imwrite(det_p, ann)
        last_msg = f"Captured #{idx}"
        last_t = time.time()
        print(f"  [{idx}] OK -> calib_{idx:02d}_{ts}_original.png")
    elif key == ord("c"):
        if cal.num_captured < 10:
            print(f"Need 10+ images, have {cal.num_captured}")
            continue
        print(f"Calibrating with {cal.num_captured} images...")
        ok = cal.calibrate()
        if ok:
            info = cal.evaluate_calibration()
            rms = info["rms_error_px"]
            cal.save()
            pre_dir = os.path.join(OUTPUT_DIR, "calib_preview")
            os.makedirs(pre_dir, exist_ok=True)
            for i in range(cal.num_captured):
                a = cal.get_annotated_image(i)
                if a is not None:
                    safe_imwrite(os.path.join(pre_dir, f"preview_{i+1:02d}_detected.png"), a)
                r = cal.get_captured_image(i)
                if r is not None:
                    safe_imwrite(os.path.join(OUTPUT_DIR, f"calib_{i+1:02d}_original.png"), r)
                a2 = cal.get_annotated_image(i)
                if a2 is not None:
                    safe_imwrite(os.path.join(OUTPUT_DIR, f"calib_{i+1:02d}_detected.png"), a2)
            print(f"CALIBRATION OK! RMS={rms:.4f}px")
            print(f"Params: {CALIB_PARAMS_FILE}")
            print(f"Preview: {pre_dir}")
        else:
            print("Calibration FAILED")
    elif key == ord("r"):
        cal.reset()
        captured = 0
        print("Reset all calibration data")
    elif key == ord("d"):
        show_ov = not show_ov
        print(f"Overlay: {show_ov}")
cv2.destroyAllWindows()
cam.stop_streaming()
cam.close()
print("Done. Exited cleanly.")
