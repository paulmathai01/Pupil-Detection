import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import math
import matplotlib.pyplot as plt
import time
import argparse
"""
img = cv2.imread('/home/akash/Desktop/eye1.jpg', 1)

cv2.imshow("image", img)
cv2.waitKey(0)
"""
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to video",)
args = vars(ap.parse_args())

def midpoint(ptA, ptB):
		return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

r = []
t = []
camera = cv2.VideoCapture(args["video"])
#camera = cv2.VideoCapture("/home/akash/Desktop/eye_sample_video/")
frameRate = camera.get(5)
while True:
	frameId = camera.get(1)
	(grabbed, img) = camera.read()	
	#img = cv2.resize(img, (480,270))
	#removing light
	if not grabbed:
		break
	x = img.copy()
	if (frameId % math.floor(frameRate) == 0):
		t.append(time.time())
		#img = img[353:530,168:468]
		#img = img[380:480,220:320]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		y = gray.copy()
	
		###		
		cv2.imshow("gray_image", gray)
		cv2.waitKey(0)
		"""
		ret,th1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
		gray = cv2.inpaint(img, th1, 5, cv2.INPAINT_TELEA)
		gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
			
		cv2.imshow("inpainted_image", gray)
		cv2.waitKey(0)
		"""
		###
		#th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            	#	cv2.THRESH_BINARY,11,2)
		###
		thresh1 = cv2.inRange(y,14,25)
		cv2.imshow("thresholded_image", thresh1)
		cv2.waitKey(0)
		###
		
		#lower_black = (120,120,120)   
		#upper_black = (245,245,245)
		#mask = cv2.inRange(gray, lower_black,upper_black)

		#cv2.imshow("image", mask)
		#cv2.waitKey(0)

		#edged = cv2.Canny(gray, 10, 20 )
		edged = cv2.dilate(thresh1, None, iterations = 5)
		edged = cv2.erode(edged, None, iterations=5)
		edged = cv2.Canny(edged, 10 ,40)
		
		cv2.imshow("Edged image", edged)
		cv2.waitKey(0)
		
		    # find contours in the edge map
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		try:
			(cnts, _) = contours.sort_contours(cnts)
			pixelsPerMetric = None

	
	    # loop over the contours individually
			for c in cnts:
				# if the contour is not sufficiently large, ignore it
				if cv2.contourArea(c) < 100:
					continue
				orig = img.copy()
				box = cv2.minAreaRect(c)
				box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
				box = np.array(box, dtype="int")

				box = perspective.order_points(box)
				cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		
				for (x, y) in box:
					cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

				(tl, tr, br, bl) = box
				(tltrX, tltrY) = midpoint(tl, tr)
				(blbrX, blbrY) = midpoint(bl, br)
				(tlblX, tlblY) = midpoint(tl, bl)
				(trbrX, trbrY) = midpoint(tr, br)

				cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
				cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
				cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
				cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

				# draw lines between the midpoints
				cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
				    (255, 0, 255), 2)
				cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
				    (255, 0, 255), 2)

				# compute the Euclidean distance between the midpoints
				dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
				dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
				if pixelsPerMetric is None:
					pixelsPerMetric = dB / 1.2

				# compute the size of the object
				dimA = dA / pixelsPerMetric
				dimB = dB / pixelsPerMetric
				r.append(min(dimA/2,dimB/2))

				# draw the object sizes on the image
				cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
				cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
				#cv2.imwrite("orig", orig)
				cv2.namedWindow("image", cv2.WINDOW_NORMAL)
				cv2.imshow("image", orig)
				cv2.waitKey(0)
				
				#IMAGE PROCESSING ENDS
			cv2.imwrite("/home/akash/Desktop/response.png", orig)

		except:
			#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
			circles = cv2.HoughCircles(y,cv2.HOUGH_GRADIENT,1,220,
				            param1=50,param2=20,minRadius=20,maxRadius=50)
			lr = []

			if circles is None:
				print ("No circles found")
			else:
			   	circles = np.uint16(np.around(circles))

			   	for i in circles[0,:]:
			      # draw the outer circle
			      		cv2.circle(x,(i[0],i[1]),i[2],(0,255,0),2)
			      # draw the center of the circle
			      		cv2.circle(x,(i[0],i[1]),2,(0,0,255),3)
			      		lr.append(i[2])
			   	rx = min(lr)/240.0
			   	r.append(rx)

			   	#cv2.imshow('detected circles',x)
				#cv2.waitKey(0)

print (len(r))
print (len(t))
plt.plot(t,r,'bo')
plt.show()
