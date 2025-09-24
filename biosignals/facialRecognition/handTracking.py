import cv2
import mediapipe as mp
import math
import database_setup	

mp_hands = mp.solutions.hands

# Helper: Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Store previous positions, totals, and "missing frame counters"
hand_data = {
    "Left": {"prev": None, "total": 0.0, "missing": 0},
    "Right": {"prev": None, "total": 0.0, "missing": 0}
}

MAX_MISSING_FRAMES = 10  # reset after N frames without detection

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,      # real-time tracking mode
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_hands = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                detected_hands.add(label)

                # Wrist landmark
                wrist = landmarks.landmark[0]
                h, w, _ = frame.shape
                wrist_pos = (int(wrist.x * w), int(wrist.y * h))

                prev = hand_data[label]["prev"]
                total = hand_data[label]["total"]

                if prev is not None:
                    step_dist = euclidean_distance(prev, wrist_pos)
                    total += step_dist
                    hand_data[label]["total"] = total

                    # Send data to InfluxDB
                    json_body = [
                        {
                            "measurement": "hand_movement",
                            "tags": {"hand": label},  # Left or Right
                            "fields": {
                                "step_distance": step_dist,
                                "total_distance": total,
                                "x_position": wrist_pos[0],
                                "y_position": wrist_pos[1]
                            }
                        }
                    ]
                    database_setup.influxdb_client.write_point(json_body)

                # Update state
                hand_data[label]["prev"] = wrist_pos
                hand_data[label]["missing"] = 0  # reset missing counter

        # Handle hands not detected this frame
        for label in hand_data.keys():
            if label not in detected_hands:
                hand_data[label]["missing"] += 1
                if hand_data[label]["missing"] >= MAX_MISSING_FRAMES:
                    hand_data[label]["prev"] = None  # reset to avoid jumps
                    hand_data[label]["missing"] = 0

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
