from __future__ import division
import mediapipe as mp
import cv2
from .eye import Eye
from .calibration import Calibration

import cv2
import mediapipe as mp


def __init__(self, calibration=None):
        self.calibration = calibration
        self.eye_left = None
        self.eye_right = None
        self.frame = None
        self.calibration = Calibration()

        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, 
                                                    max_num_faces=1, 
                                                    refine_landmarks=True, 
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)

@property
def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False



def _analyze(self):
        if self.frame is None:
            raise ValueError("Frame must be set before analysis.")

        image_h, image_w = self.frame.shape[:2]
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Convert normalized landmarks to pixel coordinates
            landmarks = [(int(lm.x * image_w), int(lm.y * image_h)) for lm in face_landmarks.landmark]

            self.eye_left = Eye(self.frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(self.frame, landmarks, 1, self.calibration)
        else:
            self.eye_left = None
            self.eye_right = None


def refresh(self, frame):
        self.frame = frame
        self._analyze()

def pupil_left_coords(self):

        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

def pupil_right_coords(self):

        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

def horizontal_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2