import math 

def get_landmark_coords(landmarks, idx, image_w, image_h):
    landmark = landmarks[idx]
    return int(landmark.x * image_w), int(landmark.y * image_h)

def find_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])