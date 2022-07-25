import cv2
import numpy as np

# Video Capture 
capture = cv2.VideoCapture(0)

# History, Threshold, DetectShadows
mog = cv2.bgsegm.createBackgroundSubtractorMOG(300)
mogg = cv2.createBackgroundSubtractorMOG2(300, 400, True)
gmg = cv2.bgsegm.createBackgroundSubtractorGMG(10, .8)
knn = cv2.createBackgroundSubtractorKNN(100, 400, True)
cnt = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

# Keeps track of what frame we're on
frameCount = 0

while(1):
	# Return Value and the current frame
	ret, frame = capture.read()

	#  Check if a current frame actually exist
	if not ret:
		break

	frameCount += 1
	# Resize the frame
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.5, fy=.50)

	# Get the foreground mask
	mogMask = mog.apply(resizedFrame)
	moggMask = mogg.apply(resizedFrame)
	gmgMask = gmg.apply(resizedFrame)
	knnMask = knn.apply(resizedFrame)
	cntMask = cnt.apply(resizedFrame)

	# Count all the non zero pixels within the mask
	mogcount = np.count_nonzero(mogMask)
	moggcount = np.count_nonzero(moggMask)
	gmgcount = np.count_nonzero(gmgMask)
	knncount = np.count_nonzero(knnMask)
	cntcount = np.count_nonzero(cntMask)

#	print('MOGFrame: %d, Pixel Count: %d' % (frameCount, mogcount))
#	print('MOG2Frame: %d, Pixel Count: %d' % (frameCount, moggcount))
#	print('GMGFrame: %d, Pixel Count: %d' % (frameCount, gmgcount))
#	print('KNNFrame: %d, Pixel Count: %d' % (frameCount, knncount))
#	print('CNTFrame: %d, Pixel Count: %d' % (frameCount, cntcount))

	cv2.putText(mogMask, 'MOG', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(moggMask, 'MOG2', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(gmgMask, 'GMG', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(knnMask, 'KNN', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(cntMask, 'CNT', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

	# Determine how many pixels do you want to detect to be considered "movement"
	if (frameCount > 1): # and count > 5000):
		if (mogcount > 5000):
			#print('Motion Detected(MOG)')
			cv2.putText(resizedFrame, 'Motion Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

		if (moggcount > 5000):
			#print('Motion Detected(MOG2)')
			cv2.putText(resizedFrame, 'Motion Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

		if (moggcount > 5000):
			#print('Motion Detected(GMG)')
			cv2.putText(resizedFrame, 'Motion Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

		if (moggcount > 5000):
			#print('Motion Detected(KNN)')
			cv2.putText(resizedFrame, 'Motion Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

		if (moggcount > 5000):
			#print('Motion Detected(CNT)')
			cv2.putText(resizedFrame, 'Motion Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

	cv2.imshow('Original', resizedFrame)
	cv2.imshow('MOG', mogMask)
	cv2.imshow('MOG2', moggMask)
	cv2.imshow('GMG', gmgMask)
	cv2.imshow('KNN', knnMask)
	cv2.imshow('CNT', cntMask)

	cv2.moveWindow('Original', 0, 0)
	cv2.moveWindow('MOG', 0, 315)
	cv2.moveWindow('MOG2',  0, 605)
	cv2.moveWindow('GMG', 719, 0)
	cv2.moveWindow('KNN', 719, 315)
	cv2.moveWindow('CNT', 719, 605)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
