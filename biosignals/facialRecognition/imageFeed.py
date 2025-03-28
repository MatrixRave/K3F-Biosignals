import cv2 as cv 

## Öffnet die Default Kamera zum Image Capturing 
camera = cv.VideoCapture(0)

## Default Width und Height der Kamera ermitteln 
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))



