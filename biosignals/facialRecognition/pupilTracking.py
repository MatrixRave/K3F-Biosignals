import cv2 as cv
import numpy as np
import recognitionVariables as recoVars
from collections import deque

ipd_mm = 63  # Average adult IPD
prev_ratios = deque(maxlen=5)  # sliding window to smooth ratio

def pupil_tracking(iris_landmarks, gray_frame, image_w, image_h, results):
    try:
        mesh_points = np.array([
            np.multiply([p.x, p.y], [image_w, image_h]).astype(int)
            for p in results.multi_face_landmarks[0].landmark
        ])

        ipd_pixels = np.linalg.norm(
            mesh_points[recoVars.leftEyeOuter] - mesh_points[recoVars.rightEyeOuter]
        )
        if ipd_pixels == 0:
            return [0, 0, np.mean(prev_ratios) if prev_ratios else 3.0]

        pixel_to_mm = ipd_mm / ipd_pixels
        (cx, cy), iris_radius_px = cv.minEnclosingCircle(mesh_points[iris_landmarks])
        iris_center = np.array([cx, cy], dtype=np.int32)
        iris_radius_mm = iris_radius_px * pixel_to_mm

        x1, y1 = max(0, iris_center[0] - int(iris_radius_px)), max(0, iris_center[1] - int(iris_radius_px))
        x2, y2 = min(image_w, iris_center[0] + int(iris_radius_px)), min(image_h, iris_center[1] + int(iris_radius_px))
        iris_roi = gray_frame[y1:y2, x1:x2]

        if iris_roi.size == 0:
            smoothed = np.mean(prev_ratios) if prev_ratios else 3.0
            return [0, iris_radius_mm, smoothed]

        blurred = cv.GaussianBlur(iris_roi, (5, 5), 0)
        _, binary = cv.threshold(blurred, 30, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        best_radius_px = 0
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse
                radius_px = (MA + ma) / 4
                area = np.pi * (MA / 2) * (ma / 2)
                if area < 5:  # small noise contour
                    continue

                # Reasonable radius bounds
                if iris_radius_px * 0.2 <= radius_px <= iris_radius_px * 0.8:
                    best_radius_px = radius_px
                    break

        if best_radius_px == 0:
            # fallback radius
            pupil_radius_mm = iris_radius_mm / 3
        else:
            pupil_radius_mm = best_radius_px * pixel_to_mm

        iris_pupil_ratio = iris_radius_mm / pupil_radius_mm if pupil_radius_mm > 0 else 0

        iris_pupil_ratio = max(1.8, min(iris_pupil_ratio, 5.2))
        prev_ratios.append(iris_pupil_ratio)
        smoothed_ratio = np.mean(prev_ratios)

        return [pupil_radius_mm, iris_radius_mm, smoothed_ratio]

    except Exception as e:
        fallback = np.mean(prev_ratios) if prev_ratios else 3.0
        return [0, 0, fallback]
