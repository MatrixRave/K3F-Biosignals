from imageFeed import camera 
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import cv2 as cv
import time 

faceDetector = FaceMeshDetector(staticMode= False, maxFaces=1, minDetectionCon= 0.5, minTrackCon= 0.5)

# Landmark Punkte des rechten Auges 
leftEyeLandmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 23]    
# Landmarks zur Normalisierung um Abstände auszugleichen 
leftLengthHor: list[int] = [159, 23]
leftLengthVert: list[int] = [130, 243]

# Landmark Punkte des rechten Auges
rightEyeLandmakrs = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
rightLengthHor: list[int] = [386, 253]
rightLengthVert: list[int] = [359, 463]

# Landmark Punkte der linken Iris
leftIrisLandmarks = [474, 475, 477, 476]  

# Landmark Punkte der rechten Iris
rightIrisLandmarks = [469, 470, 471, 472] 

# Landmark Punkte der Nase
noseLandmarks = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48, 278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274]

# Landmark Punkte des Munds 
mouthLandmakrs = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]

prev_frame_time = 0
new_frame_time = 0

plot = LivePlot(1200,480, [10,40])

while True : 

	ret, frame = camera.read()

	new_frame_time = time.time()

	# Mash wird auf das Bild der Kamera angewendet 
	frame, faces = faceDetector.findFaceMesh(frame)

	# Landmarks beider Augen einfärben 
	if faces: 
		face = faces[0]
		for landmark in leftEyeLandmarks, rightEyeLandmakrs: 
			for id in landmark:
				cv.circle(frame, face[id], 5, (255,0,255), cv.BORDER_DEFAULT)

		leftHor, _ = faceDetector.findDistance(face[leftLengthHor[0]], face[leftLengthHor[1]])
		leftVert, _ = faceDetector.findDistance(face[leftLengthVert[0]], face[leftLengthVert[1]])
		cv.line(frame, face[rightLengthHor[0]], face[rightLengthHor[1]], (255,0,0), 3)
		cv.line(frame, face[leftLengthVert[0]], face[leftLengthVert[1]], (255,0,0), 3)
		rightHor, _ = faceDetector.findDistance(face[rightLengthHor[0]], face[rightLengthHor[1]])
		rightVert, _ = faceDetector.findDistance(face[rightLengthVert[0]], face[rightLengthVert[1]])

		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = str(int(fps))


		leftBlinkRatio = (leftVert/leftHor)*10
		rightBlinkRatio = (rightVert/rightHor)*10
		blinkPlot = plot.update((leftBlinkRatio+rightBlinkRatio)/2)
		cv.imshow('BlinkPlot', blinkPlot)

		cv.putText(frame, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100,100,0), 3, cv.LINE_AA)

	# Anzeige des Bilds mit aufgelegtem Mash
	cv.imshow('FaceDetection', frame)

	# Beenden des Livefeeds
	if cv.waitKey(1) == ord('q'):
		break

#Freigabe der Ressourcen 
camera.release()
cv.destroyAllWindows()



