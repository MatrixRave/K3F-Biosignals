import time 
import matplotlib.pyplot as mtplt
import matplotlib.animation as mtpltanim

x_data, leftEyeData, rightEyeData, combinedEyesData = [],[],[],[]

fig, ax = mtplt.subplots()
leftEye, = ax.plot([], [], label='Linkes Auge')
rightEye, = ax.plot([], [], label='Rechtes Auge')
combinedEyes, = ax.plot([], [], label='Kombination')

ax.set_ylim(5,90)
ax.set_xlabel('Zeit (s)')
ax.set_ylabel('BlinkRatio')
ax.legend()

text_box = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

start_time = time.time()

def update(frame, leftBlinkRatio, rightBlinkRatio, blinkRatio, fps ): 
	current_time = time.time() - start_time

	x_data.append(current_time)
	leftEyeData.append(leftBlinkRatio)
	rightEyeData.append(rightBlinkRatio)
	combinedEyesData.append(blinkRatio)

	if len(x_data) > 500: 
		x_data.pop(0)
		leftEyeData.pop(0)
		rightEyeData.pop(0)
		combinedEyesData.pop(0)

	ax.set_xlim(max(0, current_time - 10), current_time + 1)
	leftEye.set_data(x_data, leftEyeData)
	rightEye.set_data(x_data, rightEyeData)
	combinedEyes.set_data(x_data, combinedEyesData)

	text_box.set_text(fps)

	fig.canvas.draw()
	fig.canvas.flush_events()
	mtplt.pause(0.001)

	return leftEye, rightEye, combinedEyes, text_box


def updateLivePlt():
	mtplt.ion()
	ani = mtpltanim.FuncAnimation(fig, update, interval=5000)
	mtplt.tight_layout()
	mtplt.show()
	if mtplt.waitforbuttonpress(1) & 0xFF == ord('q'):
		mtplt.close





