from collections import deque

import numpy as np
from landmarks import *
import cv2 as cv

class PupilTracking:

    def __init__(self):
        self.left_iris_radius = None
        self.right_iris_radius = None
        self.left_pupil_radius = None
        self.right_pupil_radius = None
        self.left_ratio = None
        self.right_ratio = None

        self.ipd_mm = 63  # Average adult IPD
        self.prev_ratios = deque(maxlen=5)  # sliding window to smooth ratio


    def pupil_tracking(self, gray_frame, image_w, image_h, face_landmarks):
        tracking_left = self._pupil_tracking(LEFT_IRIS, gray_frame, image_w, image_h, face_landmarks)
        tracking_right = self._pupil_tracking(RIGHT_IRIS, gray_frame, image_w, image_h, face_landmarks)

        self.left_pupil_radius, self.left_iris_radius, self.left_ratio = tracking_left
        self.right_pupil_radius, self.right_iris_radius, self.right_ratio = tracking_right

        return tracking_left + tracking_right


    def _pupil_tracking(self, iris_landmarks, gray_frame, image_w, image_h, face_landmarks):
        try:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [image_w, image_h]).astype(int)
                for p in [face_landmarks]
            ])

            ipd_pixels = np.linalg.norm(
                mesh_points[LEFT_EYE_OUTER] - mesh_points[RIGHT_EYE_OUTER]
            )
            if ipd_pixels == 0:
                return [0, 0, np.mean(self.prev_ratios) if self.prev_ratios else 3.0]

            pixel_to_mm = self.ipd_mm / ipd_pixels
            (cx, cy), iris_radius_px = cv.minEnclosingCircle(mesh_points[iris_landmarks])
            iris_center = np.array([cx, cy], dtype=np.int32)
            iris_radius_mm = iris_radius_px * pixel_to_mm

            x1, y1 = max(0, iris_center[0] - int(iris_radius_px)), max(0, iris_center[1] - int(iris_radius_px))
            x2, y2 = min(image_w, iris_center[0] + int(iris_radius_px)), min(image_h,
                                                                             iris_center[1] + int(iris_radius_px))
            iris_roi = gray_frame[y1:y2, x1:x2]

            if iris_roi.size == 0:
                smoothed = np.mean(self.prev_ratios) if self.prev_ratios else 3.0
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
            self.prev_ratios.append(iris_pupil_ratio)
            smoothed_ratio = np.mean(self.prev_ratios)

            return [pupil_radius_mm, iris_radius_mm, smoothed_ratio]

        except Exception as e:
            fallback = np.mean(self.prev_ratios) if self.prev_ratios else 3.0
            return [0, 0, fallback]


    def get_record(self):
        measurement_name = 'pupil'
        tags = {
            "bioSignal": "pupilValues"
        }
        fields = {
            "left_iris_radius": self.left_iris_radius,
            "right_iris_radius": self.right_iris_radius,
            "left_pupil_radius": self.left_pupil_radius,
            "right_pupil_radius": self.right_pupil_radius,
            "left_ratio": self.left_ratio,
            "right_ratio": self.right_ratio,
        }
        return measurement_name, tags, fields

