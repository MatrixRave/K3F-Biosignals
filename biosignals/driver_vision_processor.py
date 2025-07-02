from pupil_tracking import PupilTracking
from video_manager import VideoManager
import mediapipe as mp
from landmarks import *
from eye_blink_detection import EyeBlinkDetection
from database_setup import Database

import cv2 as cv

from influxdb_client.client.write_api import WriteApi
#from imageFeed import camera
#import mediapipe as mp
import time
#import recognitionVariables as recoVars
#import functions as func
#import databaseSetup
#import pupilTracking
import threading
#from gazeTracking.calibration import Calibration
#from gazeTracking.pupil import Pupil
import numpy as np

import sys

class DriverVisionProcessor:

    def __init__(self, video_source=0, show_gui=True, write_to_db=True):
        self.video_source = video_source
        self.video_stream = VideoManager(source=self.video_source)
        self.show_gui = show_gui
        self.fps = 0
        self.db = None

        if write_to_db:
            self.db = Database()

    def start_video_analysis(self):
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                       max_num_faces=1,
                                                       refine_landmarks=True,
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5)

        eye_blink_detector = EyeBlinkDetection(ear_threshold=0)
        blink_min_duration = 0.100
        blink_count = 0
        blink_frame_count = 0

        #pupil_tracker = PupilTracking()



        self.video_stream.open()
        while self.video_stream.is_available():
            frame_start = time.time()
            ret, frame = self.video_stream.read_frame()

            if not ret:
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            image_h, image_w = frame.shape[:2]

            frame_copy = cv.flip(frame, 1).copy()
            gray_scale = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)

            face_mesh_detection_result = mp_face_mesh.process(frame_rgb)

            if face_mesh_detection_result.multi_face_landmarks:
                face_landmarks = face_mesh_detection_result.multi_face_landmarks[0].landmark


                is_blink = eye_blink_detector.detect_eyeblink(face_landmarks, image_w, image_h)

                is_blink_event = False
                if is_blink:
                    blink_frame_count += 1
                else:
                    min_blink_frames = int(blink_min_duration * self.fps)
                    if blink_frame_count >= min_blink_frames:
                        blink_count += 1
                        sys.stdout.write('\a')
                        is_blink_event = True
                    blink_frame_count = 0


                self.db.write_record(*eye_blink_detector.get_record())
                self.db.write_record('blinks',
                                     tags = {
                                         "bioSignal": "blinkValues"
                                     },
                                     fields={
                                         "leftBlinkRatio": eye_blink_detector.eye_aspect_ratio_left,
                                         "rightBlinkRatio": eye_blink_detector.eye_aspect_ratio_right,
                                         "combinedBlinkRatio": eye_blink_detector.eye_aspect_ratio_mean,
                                         "isBlink": is_blink,
                                     })



                #pupil_tracker.pupil_tracking(gray_frame=gray_scale, image_h=image_h,
                #                             image_w=image_w, face_landmarks=face_landmarks)
                #self.db.write_record(*pupil_tracker.get_record())

                cv.putText(frame, f'FPS: {int(self.fps)}', (10, 40), cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv.LINE_AA)



            cv.imshow("FaceDetection", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            frame_end = time.time()
            self.fps = 1 / (frame_end - frame_start) if frame_end > frame_start else 0
