import cv2 as cv
import numpy as np
import recognitionVariables as recoVars

def pupil_tracking(iris_landmarks, eye_side, gray_frame, image_w, image_h, results, selected_frame):
    mesh_points = np.array([np.multiply([p.x, p.y], [image_w, image_h]).astype(int) 
                                for p in results.multi_face_landmarks[0].landmark])
    ipd_pixels = np.linalg.norm(mesh_points[recoVars.leftEyeOuter] - mesh_points[recoVars.rightEyeOuter])
    ipd_mm = 63  # Average adult IPD
    pixel_to_mm = ipd_mm / ipd_pixels

    (cx, cy), iris_radius_px = cv.minEnclosingCircle(mesh_points[iris_landmarks])

    iris_center = np.array([cx, cy], dtype=np.int32)
    iris_radius_mm = iris_radius_px * pixel_to_mm

    x1, y1 = max(0, iris_center[0] - int(iris_radius_px)), max(0, iris_center[1] - int(iris_radius_px))
    x2, y2 = min(image_w, iris_center[0] + int(iris_radius_px)), min(image_h, iris_center[1] + int(iris_radius_px))

    iris_roi = gray_frame[y1:y2, x1:x2]

    if iris_roi.size != 0:
        iris_gray = cv.equalizeHist(iris_roi)
        blurred = cv.GaussianBlur(iris_gray, (5, 5), 0)
        thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv.THRESH_BINARY_INV, 11, 2)
        edges = cv.Canny(blurred, 50, 150)
        combined = cv.bitwise_and(thresh, edges)
        contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        pupil_radius_mm = 0
        pupil_center = (int(iris_center[0]), int(iris_center[1]))

        if contours:
            largest_area = 0
            best_ellipse = None
            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv.fitEllipse(contour)
                    (x, y), (MA, ma), angle = ellipse
                    area = np.pi * (MA / 2) * (ma / 2)
                    if area > largest_area:
                        largest_area = area
                        best_ellipse = ellipse

            if best_ellipse:
                (x, y), (MA, ma), angle = best_ellipse
                radius_px = (MA + ma) / 4
                min_pupil_radius_px = iris_radius_px * 0.2
                max_pupil_radius_px = iris_radius_px * 0.7

                if min_pupil_radius_px <= radius_px <= max_pupil_radius_px:
                    pupil_radius_mm = radius_px * pixel_to_mm
                    pupil_center = (int(x), int(y))
                else:
                    pupil_radius_mm = iris_radius_mm / 3
            else:
                pupil_radius_mm = iris_radius_mm / 3

            iris_pupil_ratio = iris_radius_mm / pupil_radius_mm if pupil_radius_mm > 0 else 0
            iris_pupil_ratio = max(2.0, min(iris_pupil_ratio, 5.0))

            

            annotated_frame = selected_frame.copy()
            cv.circle(annotated_frame, tuple(iris_center), int(iris_radius_px), (0, 0, 255), 2)
            cv.circle(annotated_frame, tuple(pupil_center), int(pupil_radius_mm / pixel_to_mm), (0, 255, 0), 2)
