
import numpy as np
import cv2
from .pupil import Pupil


class Calibration:
    """
    Collects and averages binary threshold values over multiple frames
    to calibrate pupil detection based on observed iris size.
    """

    def __init__(self, nb_frames: int = 20):
        """
        Initialize the Calibration with empty threshold buffers.
        
        :param nb_frames: Number of frames needed to complete calibration per eye.
        """
        self.nb_frames = nb_frames
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self) -> bool:
        """
        Check if sufficient frames have been processed for both eyes.
        """
        return (
            len(self.thresholds_left) >= self.nb_frames
            and len(self.thresholds_right) >= self.nb_frames
        )

    def threshold(self, side: int) -> int:
        """
        Returns the average threshold for the given side (0 = left, 1 = right).
        
        :param side: 0 for left eye, 1 for right eye.
        :return: Average threshold or -1 if unavailable.
        """
        if side == 0 and self.thresholds_left:
            return int(np.mean(self.thresholds_left))
        elif side == 1 and self.thresholds_right:
            return int(np.mean(self.thresholds_right))
        return -1  # Invalid side or empty list

    @staticmethod
    def iris_size(frame: np.ndarray) -> float:
        """
        Estimate the proportion of dark pixels in the eye frame to determine iris size.
        
        :param frame: Binary image of the eye after thresholding.
        :return: Fraction of dark pixels (0.0 to 1.0).
        """
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            return 0.0  # Prevent slicing errors

        cropped = frame[5:-5, 5:-5]
        nb_pixels = cropped.size
        nb_blacks = nb_pixels - cv2.countNonZero(cropped)
        return nb_blacks / nb_pixels if nb_pixels > 0 else 0.0

    @staticmethod
    def find_best_threshold(eye_frame: np.ndarray, target_iris_size: float = 0.48) -> int:
        """
        Find the threshold that best matches the target iris size.
        
        :param eye_frame: Grayscale image of the isolated eye.
        :param target_iris_size: Expected fraction of dark pixels for the iris.
        :return: Optimal threshold value.
        """
        best_threshold = 50
        min_difference = float('inf')

        for threshold in range(5, 100, 5):
            processed = Pupil.image_processing(eye_frame, threshold)
            size = Calibration.iris_size(processed)
            difference = abs(size - target_iris_size)

            if difference < min_difference:
                min_difference = difference
                best_threshold = threshold

        return best_threshold

    def evaluate(self, eye_frame: np.ndarray, side: int):
        """
        Evaluate and store the optimal threshold for a given eye frame and side.

        :param eye_frame: Grayscale image of the eye.
        :param side: 0 for left, 1 for right.
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
