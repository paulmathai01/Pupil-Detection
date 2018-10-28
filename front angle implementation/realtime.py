import cv2
import numpy as np
import mahotas
from PIL import Image
from PIL import ImageEnhance
import argparse
from imutils.video import VideoStream
from imutils.video import FPS
import time


def adjust_sharpness(imgIn):
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0 
    boxFilter = np.ones( (9,9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(imgIn, -1, kernel)
    return(custom)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


a=0
while True:
    a=a+1
    frame = vs.read()

    frame = cv2.resize(frame,(480,270))
    print("[INFO] Resizing")
    M = np.ones(frame.shape,dtype = "uint8")*50
    added = cv2.add(frame,M)
    img = cv2.cvtColor(added,cv2.COLOR_BGR2GRAY)
    print("[INFO] Converted from RGB to GRAY for interation=",a)
    img = cv2.equalizeHist(img)
    #img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.medianBlur(img,1)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    
    
    
    #edged = cv2.dilate(img,None,iterations = 4)
    #edged = cv2.erode(edged,None,iterations = 4)
    #img = adjust_sharpness(img)
    print("[INFO] Eroded and Sharpened")

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,.5,15,param1=30,param2=100,minRadius=0,maxRadius=200)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            print("[INFO] Computing for (",x,",",y,",",r,")")
    cv2.imshow("Added",img)
    key = cv2.waitKey(1) & 0xFF
        
    if key == ord("q"):
        break
    fps.update()
        
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
            

