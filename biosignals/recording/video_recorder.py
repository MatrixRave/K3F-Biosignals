import cv2
import time
import csv
import os
import datetime as dt

def record_video(src, out_dir, target_fps):
    codec = "MJPG"

    out_dir = os.path.join(out_dir, f"r3_capture_{int(target_fps)}fps_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "capture.avi")
    csv_path = os.path.join(out_dir, "timestamps.csv")

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps_reported = cap.get(cv2.CAP_PROP_FPS) or target_fps

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_idx","mono_ns","utc_iso","utc_epoch_ns","width","height"])
        start_info = {
            "capture_started_utc": dt.datetime.utcnow().isoformat(timespec="milliseconds")+"Z",
            "fps_target": target_fps,
            "fps_reported": fps_reported,
            "codec": codec,
            "size": f"{w}x{h}",
        }
        wr.writerow([f"#meta {start_info}"])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mono_ns = time.perf_counter_ns()
            utc = dt.datetime.utcnow().isoformat(timespec="milliseconds")+"Z"
            utc_epoch_ns = time.time_ns()

            writer.write(frame)
            wr.writerow([frame_idx, mono_ns, utc, utc_epoch_ns, frame.shape[1], frame.shape[0]])

            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done: {video_path}\nTimestamps: {csv_path}")


def run():
    src = 0
    out_dir = "./video-recording"
    target_fps = 30
    record_video(src, out_dir, target_fps)


if __name__ == "__main__":
    run()