from influxdb_client.client.write_api import WriteApi
from imageFeed import camera
import cv2 as cv
import mediapipe as mp
import time
import recognitionVariables as recoVars
import functions as func
import databaseSetup
import pupilTracking
import threading
from gazeTracking.calibration import Calibration
from gazeTracking.pupil import Pupil
import numpy as np

# MediaPipe Initialisierung
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

write_api: WriteApi = databaseSetup.client.write_api()

thread_id = threading.get_native_id()

calibration = Calibration()
calibration_frame_count = 0
max_calibration_frames = calibration.nb_frames

while True:
    ret, frame = camera.read()
    if not ret:
        break

    recoVars.new_frame_time = time.time()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    image_h, image_w = frame.shape[:2]

    frame_copy = cv.flip(frame,1).copy() 
    gray_scale = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        for landmark_set in [recoVars.leftEyeLandmarks, recoVars.rightEyeLandmarks]:
            for id in landmark_set:
                x, y = func.get_landmark_coords(landmarks, id, image_w, image_h)
                cv.circle(frame, (x, y), 2, (255, 0, 255), cv.FILLED)

        # Linkes Auge
        p1 = func.get_landmark_coords(landmarks, recoVars.leftLengthHor[0], image_w, image_h)
        p2 = func.get_landmark_coords(landmarks, recoVars.leftLengthHor[1], image_w, image_h)
        leftHor = func.find_distance(p1, p2)
        cv.line(frame, p1, p2, (255, 0, 0), 2)

        p3 = func.get_landmark_coords(landmarks, recoVars.leftLengthVert[0], image_w, image_h)
        p4 = func.get_landmark_coords(landmarks, recoVars.leftLengthVert[1], image_w, image_h)
        leftVert = func.find_distance(p3, p4)
        cv.line(frame, p3, p4, (255, 0, 0), 2)

        # Rechtes Auge
        p5 = func.get_landmark_coords(landmarks, recoVars.rightLengthHor[0], image_w, image_h)
        p6 = func.get_landmark_coords(landmarks, recoVars.rightLengthHor[1], image_w, image_h)
        rightHor = func.find_distance(p5, p6)
        cv.line(frame, p5, p6, (0, 255, 0), 2)

        p7 = func.get_landmark_coords(landmarks, recoVars.rightLengthVert[0], image_w, image_h)
        p8 = func.get_landmark_coords(landmarks, recoVars.rightLengthVert[1], image_w, image_h)
        rightVert = func.find_distance(p7, p8)
        cv.line(frame, p7, p8, (0, 255, 0), 2)

        # Blink-Ratio
        leftBlinkRatio = max((leftVert / leftHor) * 10, 0)
        rightBlinkRatio = max((rightVert / rightHor) * 10, 0)
        blinkRatio = max((leftBlinkRatio + rightBlinkRatio) / 2, 0)

        # --- Extract cropped eye regions for calibration ---
        def extract_eye_crop(landmark_ids):
            points = np.array([
                func.get_landmark_coords(landmarks, id, image_w, image_h)
                for id in landmark_ids
            ])
            mask = np.zeros(gray_scale.shape, dtype=np.uint8)
            cv.fillPoly(mask, [points], 255)
            eye = cv.bitwise_and(gray_scale, gray_scale, mask=mask)
            x, y, w, h = cv.boundingRect(points)
            return eye[y:y+h, x:x+w]
        
        left_eye = Eye(gray_scale, landmarks, 0, calibration)
        right_eye = Eye(gray_scale, landmarks, 1, calibration)

        if left_eye.pupil and right_eye.pupil:
            # Draw pupils
            cv.circle(frame, (left_eye.pupil.x + left_eye.origin[0], left_eye.pupil.y + left_eye.origin[1]), 3, (0, 255, 255), -1)
            cv.circle(frame, (right_eye.pupil.x + right_eye.origin[0], right_eye.pupil.y + right_eye.origin[1]), 3, (0, 255, 255), -1)

            # Gaze ratio (pupil displacement from eye center)
            def get_gaze_ratio(pupil, center):
                dx = pupil.x - center[0]
                dy = pupil.y - center[1]
                return dx / center[0], dy / center[1]

            left_gaze = get_gaze_ratio(left_eye.pupil, left_eye.center)
            right_gaze = get_gaze_ratio(right_eye.pupil, right_eye.center)

            # Optional: Draw gaze direction or print
            print(f"Gaze - L: {left_gaze}, R: {right_gaze}")



          # FPS anzeigen
        fps = 1 / (recoVars.new_frame_time - recoVars.prev_frame_time)
        recoVars.prev_frame_time = recoVars.new_frame_time

        write_api.write(bucket=databaseSetup.bucket, org=databaseSetup.org, record=databaseSetup.create_payload_blink(leftBlinkRatio= leftBlinkRatio, rightBlinkRatio= rightBlinkRatio, combinedBlinkRatio=blinkRatio, threadId = thread_id,) )
        

        leftPupil = pupilTracking.pupil_tracking(gray_frame= gray_scale, image_h= image_h, image_w = image_w, iris_landmarks=recoVars.leftIris, results=results)
        rightPupil = pupilTracking.pupil_tracking(gray_frame= gray_scale, image_h= image_h, image_w = image_w, iris_landmarks=recoVars.rightIris, results=results)

        write_api.write(bucket=databaseSetup.bucket, org=databaseSetup.org, 
                        record=databaseSetup.create_payload_pupil(left_iris_radius=leftPupil[1], left_pupil_radius=leftPupil[0], left_ratio= leftPupil[2], 
                                                                  right_iris_radius= rightPupil[1], right_pupil_radius=rightPupil[0], right_ratio= rightPupil[2], threadId = thread_id,))

        cv.putText(frame, f'FPS: {int(fps)}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv.LINE_AA)
        
    cv.imshow("FaceDetection", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
   

write_api.flush
camera.release()
cv.destroyAllWindows()
write_api.close()
databaseSetup.client.close()
