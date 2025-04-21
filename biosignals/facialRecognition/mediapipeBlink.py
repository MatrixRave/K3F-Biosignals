from imageFeed import camera
import cv2 as cv
import mediapipe as mp
import time
import math
from cvzone.PlotModule import LivePlot
import plotting as plt

# MediaPipe Initialisierung
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Zeichnen optional
mp_drawing = mp.solutions.drawing_utils

# Gesichtslandmark-IDs
leftEyeLandmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 23]
leftLengthHor = [159, 23]
leftLengthVert = [130, 243]

rightEyeLandmarks = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
rightLengthHor = [386, 253]
rightLengthVert = [359, 463]

plot = LivePlot(1200, 480, [10, 40])

prev_frame_time = 0
new_frame_time = 0

def get_landmark_coords(landmarks, idx, image_w, image_h):
    landmark = landmarks[idx]
    return int(landmark.x * image_w), int(landmark.y * image_h)

def find_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = camera.read()
    if not ret:
        break

    new_frame_time = time.time()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    image_h, image_w = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        for landmark_set in [leftEyeLandmarks, rightEyeLandmarks]:
            for id in landmark_set:
                x, y = get_landmark_coords(landmarks, id, image_w, image_h)
                cv.circle(frame, (x, y), 2, (255, 0, 255), cv.FILLED)

        # Linkes Auge
        p1 = get_landmark_coords(landmarks, leftLengthHor[0], image_w, image_h)
        p2 = get_landmark_coords(landmarks, leftLengthHor[1], image_w, image_h)
        leftHor = find_distance(p1, p2)
        cv.line(frame, p1, p2, (255, 0, 0), 2)

        p3 = get_landmark_coords(landmarks, leftLengthVert[0], image_w, image_h)
        p4 = get_landmark_coords(landmarks, leftLengthVert[1], image_w, image_h)
        leftVert = find_distance(p3, p4)
        cv.line(frame, p3, p4, (255, 0, 0), 2)

        # Rechtes Auge
        p5 = get_landmark_coords(landmarks, rightLengthHor[0], image_w, image_h)
        p6 = get_landmark_coords(landmarks, rightLengthHor[1], image_w, image_h)
        rightHor = find_distance(p5, p6)
        cv.line(frame, p5, p6, (0, 255, 0), 2)

        p7 = get_landmark_coords(landmarks, rightLengthVert[0], image_w, image_h)
        p8 = get_landmark_coords(landmarks, rightLengthVert[1], image_w, image_h)
        rightVert = find_distance(p7, p8)
        cv.line(frame, p7, p8, (0, 255, 0), 2)

        # Blink-Ratio
        leftBlinkRatio = (leftVert / leftHor) * 10
        rightBlinkRatio = (rightVert / rightHor) * 10
        blinkRatio = (leftBlinkRatio + rightBlinkRatio) / 2

          # FPS anzeigen
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        plt.update(frame, leftBlinkRatio = leftBlinkRatio, rightBlinkRatio = rightBlinkRatio, blinkRatio = blinkRatio, fps = fps)
        
    cv.imshow("FaceDetection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv.destroyAllWindows()

