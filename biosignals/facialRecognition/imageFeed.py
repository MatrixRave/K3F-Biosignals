import cv2 

## Öffnet die Default Kamera zum Image Capturing 
camera = cv2.VideoCapture(0)

## Default Width und Height der Kamera ermitteln 
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))



