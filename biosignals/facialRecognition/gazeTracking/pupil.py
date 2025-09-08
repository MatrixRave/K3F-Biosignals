import numpy as np
import cv2

class Pupil:
    def __init__(self, eye_frame: np.ndarray, threshold: int):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame: np.ndarray, threshold: int) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        filtered = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        eroded = cv2.erode(filtered, kernel, iterations=2)
        _, binary = cv2.threshold(eroded, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def detect_iris(self, eye_frame: np.ndarray):
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        contours, _ = cv2.findContours(
            self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) < 1:
            return  # No contours found

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            if len(contour) >= 5:  # avoid tiny noisy contours
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    self.x = int(M["m10"] / M["m00"])
                    self.y = int(M["m01"] / M["m00"])
                    return  # Stop after first valid detection
