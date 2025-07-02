from landmarks import *
import utils
import sys

class EyeBlinkDetection:

    def __init__(self, ear_threshold):
        self.blink_lower_threshold = 0.22 * 10
        self.blink_upper_threshold = 0.25 * 10
        self.blink_confidence = 0.50

        self.eye_aspect_ratio_left = None
        self.eye_aspect_ratio_right = None
        self.eye_aspect_ratio_mean = None

        pass

    
    def detect_eyeblink(self, face_landmarks, image_w, image_h):
        # Linkes Auge
        p1 = utils.get_landmark_coords(face_landmarks, LEFT_LENGTH_HOR[0], image_w, image_h)
        p2 = utils.get_landmark_coords(face_landmarks, LEFT_LENGTH_HOR[1], image_w, image_h)
        leftHor = utils.find_distance(p1, p2)
        #cv.line(frame, p1, p2, (255, 0, 0), 2)

        p3 = utils.get_landmark_coords(face_landmarks, LEFT_LENGTH_VERT[0], image_w, image_h)
        p4 = utils.get_landmark_coords(face_landmarks, LEFT_LENGTH_VERT[1], image_w, image_h)
        leftVert = utils.find_distance(p3, p4)
        #cv.line(frame, p3, p4, (255, 0, 0), 2)

        # Rechtes Auge
        p5 = utils.get_landmark_coords(face_landmarks, RIGHT_LENGTH_HOR[0], image_w, image_h)
        p6 = utils.get_landmark_coords(face_landmarks, RIGHT_LENGTH_HOR[1], image_w, image_h)
        rightHor = utils.find_distance(p5, p6)
        #cv.line(frame, p5, p6, (0, 255, 0), 2)

        p7 = utils.get_landmark_coords(face_landmarks, RIGHT_LENGTH_VERT[0], image_w, image_h)
        p8 = utils.get_landmark_coords(face_landmarks, RIGHT_LENGTH_VERT[1], image_w, image_h)
        rightVert = utils.find_distance(p7, p8)
        #cv.line(frame, p7, p8, (0, 255, 0), 2)

        # Blink-Ratio
        self.eye_aspect_ratio_left = max((leftVert / leftHor) * 10, 0)
        self.eye_aspect_ratio_right = max((rightVert / rightHor) * 10, 0)
        self.eye_aspect_ratio_mean = max((self.eye_aspect_ratio_left + self.eye_aspect_ratio_right) / 2, 0)


        if (self.eye_aspect_ratio_mean > self.blink_lower_threshold
                and self.eye_aspect_ratio_mean <= self.blink_upper_threshold):
            print(
                "I think person blinked. eye_aspect_ratio_mean = ",
                self.eye_aspect_ratio_mean,
                "Confirming with ViT model...",
            )
            is_eyeblink = True
            #is_eyeblink = self.blink_detection_model(left_eye=left_eye, right_eye=right_eye)
            if is_eyeblink:
                print("Yes, person blinked. Confirmed by model")

            else:
                print("No, person didn't blinked. False Alarm")
        elif self.eye_aspect_ratio_mean <= self.blink_lower_threshold:
            is_eyeblink = True
            print("Surely person blinked. eye_aspect_ratio_mean = ", self.eye_aspect_ratio_mean)

        else:
            is_eyeblink = False

        return is_eyeblink


    def get_record(self):
        measurement_name = 'blinks'
        tags = {
            "bioSignal": "blinkValues"
        }
        fields = {
            "leftBlinkRatio": self.eye_aspect_ratio_left,
            "rightBlinkRatio": self.eye_aspect_ratio_right,
            "combinedBlinkRatio": self.eye_aspect_ratio_mean
        }
        return measurement_name, tags, fields



        