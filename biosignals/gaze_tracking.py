import numpy as np

import utils
import cv2 as cv

class GazeTracking():

    def __init__(self):
        pass

    def get_gaze_ratio(pupil, center):
        dx = pupil.x - center[0]
        dy = pupil.y - center[1]
        return dx / center[0], dy / center[1]


    def extract_eye_crop(self, landmark_ids):
        points = np.array([
            utils.get_landmark_coords(landmarks, id, image_w, image_h)
            for id in landmark_ids
        ])
        mask = np.zeros(gray_scale.shape, dtype=np.uint8)
        cv.fillPoly(mask, [points], 255)
        eye = cv.bitwise_and(gray_scale, gray_scale, mask=mask)
        x, y, w, h = cv.boundingRect(points)
        return eye[y:y + h, x:x + w]

    def gaze_tracking(self):
        # --- Extract cropped eye regions for calibration ---

        left_eye = Eye(gray_scale, landmarks, 0, calibration)
        right_eye = Eye(gray_scale, landmarks, 1, calibration)

        if left_eye.pupil and right_eye.pupil:
            # Draw pupils
            cv.circle(frame, (left_eye.pupil.x + left_eye.origin[0], left_eye.pupil.y + left_eye.origin[1]), 3,
                      (0, 255, 255), -1)
            cv.circle(frame, (right_eye.pupil.x + right_eye.origin[0], right_eye.pupil.y + right_eye.origin[1]),
                      3, (0, 255, 255), -1)

            # Gaze ratio (pupil displacement from eye center)


            left_gaze = get_gaze_ratio(left_eye.pupil, left_eye.center)
            right_gaze = get_gaze_ratio(right_eye.pupil, right_eye.center)

            # Optional: Draw gaze direction or print
            print(f"Gaze - L: {left_gaze}, R: {right_gaze}")


    def get_record(self):
        pass