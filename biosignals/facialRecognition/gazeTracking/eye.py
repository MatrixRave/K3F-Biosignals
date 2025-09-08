import numpy as np
import cv2
from .pupil import Pupil
from recognitionVariables import leftEyeLandmarks, rightEyeLandmarks

class Eye:
    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """
        Extracts and isolates the eye region from the original frame using landmark indices.
        """
        region = np.array([landmarks[point] for point in points], dtype=np.int32)
        self.landmark_points = region

        height, width = frame.shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        cv2.fillPoly(mask, [region], 0)
        eye = cv2.bitwise_not(np.zeros_like(frame), frame.copy(), mask=mask)

        # Cropping
        margin = 5
        min_x = max(np.min(region[:, 0]) - margin, 0)
        max_x = min(np.max(region[:, 0]) + margin, width)
        min_y = max(np.min(region[:, 1]) - margin, 0)
        max_y = min(np.max(region[:, 1]) + margin, height)

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        if self.frame.size != 0:
            h, w = self.frame.shape[:2]
            self.center = (w / 2, h / 2)
        else:
            self.center = (0, 0)

    def _analyze(self, original_frame, landmarks, side, calibration):
        """
        Analyzes the eye from frame and landmark data.
        """
        if side == 0:
            points = leftEyeLandmarks  # define this globally
        elif side == 1:
            points = rightEyeLandmarks  # define this globally
        else:
            return

        self._isolate(original_frame, landmarks, points)

        if self.frame is None or self.frame.size == 0:
            return  # Prevent crashing on empty frame

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
