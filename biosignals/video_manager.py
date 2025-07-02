import cv2
import time
import os

from numpy.ma.core import true_divide


class VideoManager:
    def __init__(self, source=0, output_path=None):
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.fps = 0.0
        self.frame_size = None
        self.is_recording = False

    @staticmethod
    def list_cameras(max_cameras=10):
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                available.append(i)
            cap.release()
        return available

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Quelle '{self.source}' konnte nicht geöffnet werden.")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (width, height)

    def start_recording(self, fps=30.0):
        if not self.output_path:
            raise ValueError("Kein Ausgabe-Pfad für das Video angegeben.")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # mp4v, Oder 'XVID' für .avi
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, self.frame_size)
        self.is_recording = True

    def stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stop_recording()
        cv2.destroyAllWindows()

    def is_available(self):
        return self.cap.isOpened()

    def read_frame(self):
        if not self.cap:
            raise RuntimeError("Quelle ist nicht geöffnet.")
        start = time.time()
        ret, frame = self.cap.read()
        end = time.time()
        self.fps = 1 / (end - start) if end > start else 0
        return ret, frame

    def stream(self, show_fps=True):
        self.open()
        try:
            while True:
                ret, frame = self.read_frame()
                if frame is None:
                    break

                if show_fps:
                    cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if self.is_recording and self.writer:
                    self.writer.write(frame)

                cv2.imshow("VideoManager", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            self.close()
