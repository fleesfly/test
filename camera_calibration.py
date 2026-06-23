import sys, cv2, numpy as np, os, json
from typing import Tuple, List, Optional, Dict
sys.stdout.reconfigure(encoding="utf-8")

from config import CALIB_CFG, CALIB_PARAMS_FILE, save_calibration_params, load_calibration_params


class HalconDotCalibrator:
    def __init__(self, config=None):
        self.cfg = config or CALIB_CFG
        self.grid_size = (self.cfg.dot_grid_cols, self.cfg.dot_grid_rows)
        self.num_dots = self.cfg.dot_grid_rows * self.cfg.dot_grid_cols
        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self.image_shapes: List[Tuple] = []
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self._calib_frames: List[np.ndarray] = []
        self._calib_annotated: List[np.ndarray] = []
        self._calib_detected_centers: List[np.ndarray] = []
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.rms_error: float = float("inf")
        self._per_view_errors: List[float] = []
        self._sorted_view_indices: List[int] = []
        self._image_paths: List[str] = []
        self.object_point_template = self._build_object_point_template()

    def _build_object_point_template(self) -> np.ndarray:
        cols, rows = self.cfg.dot_grid_cols, self.cfg.dot_grid_rows
        spacing = self.cfg.dot_spacing_mm
        points = np.zeros((rows * cols, 3), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                points[idx] = [(c - (cols - 1) / 2.0) * spacing,
                               (r - (rows - 1) / 2.0) * spacing, 0.0]
        return points

    def _auto_preprocess(self, gray: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(gray, [2, 98])
        if hi - lo > 10:
            stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            stretched = gray.copy()
        mean_contrast = np.std(stretched)
        clip = max(1.0, min(4.0, 80.0 / max(mean_contrast, 1)))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(stretched)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        return blurred

    def detect_dots(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image[:,:,0].copy()
        else:
            gray = image.copy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            annotated = image.copy()
        else:
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        processed = self._auto_preprocess(gray)
        scales = [1.0, 0.8, 1.2]
        for scale in scales:
            if scale != 1.0:
                h, w = processed.shape
                scaled = cv2.resize(processed, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                scaled = processed
            success, centers = self._detect_circles_grid(scaled, scale)
            if success and centers is not None and len(centers) == self.num_dots:
                refined = cv2.cornerSubPix(
                    gray, centers,
                    self.cfg.subpix_win_size, self.cfg.subpix_zero_zone,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     self.cfg.subpix_max_iter, self.cfg.subpix_epsilon))
                self._draw_dots(annotated, refined)
                return True, refined, annotated
        return self._detect_dots_hough_fallback(gray, processed, annotated)

    def _detect_circles_grid(self, processed: np.ndarray, scale: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        methods = [
            (cv2.CALIB_CB_SYMMETRIC_GRID, "SYMMETRIC"),
            (cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING, "SYMMETRIC+CLUSTER"),
        ]
        for flags, name in methods:
            try:
                success, centers = cv2.findCirclesGrid(processed, self.grid_size, None, flags)
                if success:
                    if scale != 1.0:
                        centers = centers / scale
                    return True, centers
            except:
                pass
        print(f"[Calib] findCirclesGrid failed at scale={scale:.1f}, try Hough fallback")
        return False, None

    def _preprocess_for_dots(self, gray: np.ndarray) -> np.ndarray:
        return self._auto_preprocess(gray)

    def _detect_dots_hough_fallback(self, gray: np.ndarray, processed: np.ndarray,
                                     annotated: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        circles = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT,
                                     dp=self.cfg.hough_dp,
                                     minDist=self.cfg.hough_min_dist,
                                     param1=self.cfg.hough_param1,
                                     param2=self.cfg.hough_param2,
                                     minRadius=self.cfg.min_radius,
                                     maxRadius=self.cfg.max_radius)
        if circles is not None:
            print(f"[Calib] HoughFallback: 检测到 {circles.shape[1]} 个圆, 半径: {circles[0][:,2].min():.1f}~{circles[0][:,2].max():.1f}")
        if circles is None:
            return False, None, annotated
        centers = circles[0, :, :2].astype(np.float32)
        if len(centers) < self.num_dots:
            return False, None, annotated
        centers = self._filter_and_sort_centers(centers)
        if len(centers) < self.num_dots:
            return False, None, annotated
        centers = centers[:self.num_dots]
        refined = cv2.cornerSubPix(gray, centers,
                                     self.cfg.subpix_win_size, self.cfg.subpix_zero_zone,
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                      self.cfg.subpix_max_iter, self.cfg.subpix_epsilon))
        self._draw_dots(annotated, refined)
        return True, refined, annotated

    def _filter_and_sort_centers(self, centers: np.ndarray) -> np.ndarray:
        cen = np.array(centers)
        if len(cen) < 4:
            return cen
        centroid = cen.mean(axis=0)
        angles = np.arctan2(cen[:, 1] - centroid[1], cen[:, 0] - centroid[0])
        order = np.argsort(angles)
        return cen[order]

    def _draw_dots(self, img: np.ndarray, centers: np.ndarray):
        for c in centers:
            pt = (int(round(c[0])), int(round(c[1])))
            cv2.circle(img, pt, 4, (0, 0, 255), -1)
            cv2.circle(img, pt, 6, (0, 255, 255), 1)

    def add_image(self, image: np.ndarray, filepath: str = "") -> bool:
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image[:,:,0].copy()
        else:
            gray = image.copy()
        success, centers, annotated = self.detect_dots(gray)
        if success:
            self.object_points.append(self.object_point_template.copy())
            self.image_points.append(centers)
            self.image_shapes.append(gray.shape[::-1])
            self._calib_frames.append(gray)
            self._calib_annotated.append(annotated)
            self._calib_detected_centers.append(centers)
            self._image_paths.append(filepath)
            print(f"[Calib] Added: {len(self.object_points)}/{self.cfg.min_calib_images}")
            return True
        return False


    def add_calibration_image(self, image: np.ndarray, centers=None, annotated=None, filepath: str = "") -> bool:
        """兼容层：接收已检测好的 centers/annotated，或回退到 add_image"""
        if centers is not None:
            if len(image.shape) == 2:
                gray = image.copy()
            elif image.shape[2] == 1:
                gray = image[:,:,0].copy()
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.object_points.append(self.object_point_template.copy())
            self.image_points.append(centers)
            self.image_shapes.append(gray.shape[::-1])
            self._calib_frames.append(gray)
            self._calib_annotated.append(annotated if annotated is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            self._calib_detected_centers.append(centers)
            self._image_paths.append(filepath)
            print(f"[Calib] Added: {len(self.object_points)}/{self.cfg.min_calib_images}")
            return True
        return self.add_image(image, filepath)


    
    def calibrate(self) -> Tuple[bool, float]:
        n = len(self.object_points)
        if n < 4:
            return False, 0.0
        img_size = self.image_shapes[0]
        calib_flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS |
            cv2.CALIB_RATIONAL_MODEL |
            cv2.CALIB_THIN_PRISM_MODEL
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        try:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, img_size,
                None, None, criteria=criteria, flags=calib_flags)
        except:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, img_size,
                None, None)
        self.camera_matrix = K
        self.dist_coeffs = D
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.rms_error = rms
        self._compute_per_view_errors()
        self._compute_uncertainty()
        print(f"[Calib] RMS={rms:.4f}px | {n} images | fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
        return True, rms

    def _compute_per_view_errors(self):
        errors = []
        for i in range(len(self.object_points)):
            proj, _ = cv2.projectPoints(
                self.object_points[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.dist_coeffs)
            err = np.sqrt(np.mean(np.square(self.image_points[i] - proj[:, 0, :])))
            errors.append(err)
        self._per_view_errors = errors
        self._sorted_view_indices = np.argsort(errors).tolist()

    def _compute_uncertainty(self):
        n = len(self.object_points)
        if n < 4:
            self._fx_stderr = self._fy_stderr = 0
            return
        fx_vals, fy_vals = [], []
        for i in range(n):
            obj_pts = self.object_points[:i] + self.object_points[i+1:]
            img_pts = self.image_points[:i] + self.image_points[i+1:]
            shapes = self.image_shapes[:i] + self.image_shapes[i+1:]
            try:
                _, K, _, _, _ = cv2.calibrateCamera(
                    obj_pts, img_pts, shapes[0], None, None,
                    flags=cv2.CALIB_USE_INTRINSIC_GUESS)
                fx_vals.append(K[0, 0])
                fy_vals.append(K[1, 1])
            except:
                pass
        self._fx_stderr = float(np.std(fx_vals)) if fx_vals else 0
        self._fy_stderr = float(np.std(fy_vals)) if fy_vals else 0

    def evaluate_calibration(self) -> Dict:
        K = self.camera_matrix
        D = self.dist_coeffs
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        k1, k2, p1, p2, k3 = D.ravel()[:5] if D is not None else (0,)*5
        extrinsics_summary = []
        errors = self._per_view_errors
        if self.rvecs:
            for i in range(len(self.rvecs)):
                rmat, _ = cv2.Rodrigues(self.rvecs[i])
                angles = cv2.RQDecomp3x3(rmat)[0]
                t = self.tvecs[i].ravel()
                extrinsics_summary.append({
                    "image": i,
                    "angles_deg": [round(float(a), 1) for a in angles],
                    "translation_mm": [round(float(t[j]), 1) for j in range(3)],
                    "error_px": round(float(errors[i]), 4),
                })
        coverage = self._analyze_coverage()
        return {
            "rms_error_px": round(float(self.rms_error), 4),
            "camera_matrix": K.tolist(),
            "focal_length": (round(float(fx), 1), round(float(fy), 1)),
            "principal_point": (round(float(cx), 1), round(float(cy), 1)),
            "distortion_coefficients": {
                "k1": round(k1, 6), "k2": round(k2, 6),
                "p1": round(p1, 6), "p2": round(p2, 6), "k3": round(k3, 6),
            },
            "fx_stderr": getattr(self, "_fx_stderr", 0),
            "fy_stderr": getattr(self, "_fy_stderr", 0),
            "image_size": list(self.image_shapes[0][::-1]) if self.image_shapes else [0, 0],
            "num_images": len(self.object_points),
            "per_view_errors": [round(e, 4) for e in errors],
            "best_index": self._sorted_view_indices[0] if self._sorted_view_indices else 0,
            "worst_index": self._sorted_view_indices[-1] if self._sorted_view_indices else 0,
            "coverage": coverage,
            "extrinsics": extrinsics_summary,
            "improvement_hints": self.get_calibration_improvement_hints(),
        }

    def _analyze_coverage(self) -> Dict:
        if not self.image_points:
            return {}
        all_pts = np.vstack(self.image_points)
        x, y = all_pts[:, 0], all_pts[:, 1]
        w, h = self.image_shapes[0] if self.image_shapes else (640, 480)
        return {
            "x_range": (float(x.min()), float(x.max())),
            "y_range": (float(y.min()), float(y.max())),
            "coverage_x_pct": round(float((x.max() - x.min()) / w * 100), 1),
            "coverage_y_pct": round(float((y.max() - y.min()) / h * 100), 1),
            "num_unique_views": len(self.object_points),
        }

    def save(self, filepath: str = CALIB_PARAMS_FILE):
        if self.camera_matrix is None:
            raise ValueError("No calibration performed")
        img_size = self.image_shapes[0] if self.image_shapes else (0, 0)
        extrinsics_data = []
        if self.rvecs is not None:
            for i in range(len(self.rvecs)):
                extrinsics_data.append({
                    "rvec": self.rvecs[i].ravel().tolist(),
                    "tvec": self.tvecs[i].ravel().tolist(),
                    "error_px": round(float(self._per_view_errors[i]), 4),
                })
        save_calibration_params(self.camera_matrix, self.dist_coeffs,
                                 self.rms_error, img_size, filepath,
                                 extrinsics=extrinsics_data)
        print(f"[Calib] Saved: {filepath}")

    def load(self, filepath: str = CALIB_PARAMS_FILE) -> bool:
        params = load_calibration_params(filepath)
        if params is None:
            return False
        self.camera_matrix = params["camera_matrix"]
        self.dist_coeffs = params["dist_coeffs"]
        self.rms_error = params["rms_error"]
        return True

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None:
            return image
        h, w = image.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_mtx)
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]
        return undistorted

    def get_calibration_improvement_hints(self) -> List[str]:
        hints = []
        if not self._per_view_errors:
            return hints
        errors = np.array(self._per_view_errors)
        if errors.max() > errors.mean() * 2:
            worst = int(np.argmax(errors))
            hints.append(f"Image #{worst} error ({errors[worst]:.2f}px) too large, consider removing")
        if self._fx_stderr > 50:
            hints.append(f"Focal length uncertainty high (std={self._fx_stderr:.1f}), add more images")
        coverage = self._analyze_coverage()
        if coverage.get("coverage_x_pct", 100) < 50:
            hints.append("Board coverage insufficient horizontally, move to left/right edges")
        if coverage.get("coverage_y_pct", 100) < 50:
            hints.append("Board coverage insufficient vertically, move to top/bottom edges")
        if len(self.object_points) < 15:
            hints.append(f"Collect more images ({len(self.object_points)} current, 15+ recommended)")
        return hints

    def reset(self):
        self.object_points.clear()
        self.image_points.clear()
        self.image_shapes.clear()
        self._calib_frames.clear()
        self._calib_annotated.clear()
        self._calib_detected_centers.clear()
        self._image_paths.clear()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = float("inf")
        self._per_view_errors.clear()
        self._sorted_view_indices.clear()

    @property
    def num_captured(self) -> int:
        return len(self.object_points)

    def get_captured_image(self, index: int) -> Optional[np.ndarray]:
        return self._calib_frames[index] if 0 <= index < len(self._calib_frames) else None

    def get_annotated_image(self, index: int) -> Optional[np.ndarray]:
        return self._calib_annotated[index] if 0 <= index < len(self._calib_annotated) else None

if __name__ == '__main__':
    c = HalconDotCalibrator()
    if c.load():
        print(json.dumps(c.evaluate_calibration(), indent=2, ensure_ascii=False))
        hints = c.get_calibration_improvement_hints()
        if hints:
            print()
            print('Optimization suggestions:')
            for h in hints:
                print(f'  - {h}')
    else:
        print('No calibration params found')
