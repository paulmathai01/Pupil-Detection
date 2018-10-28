import cv2
import numpy as np
import mahotas
from PIL import Image
from PIL import ImageEnhance
import argparse


def adjust_sharpness(imgIn):
    kernel = np.zeros( (9,9), np.float32)
    kernel[4,4] = 2.0 
    boxFilter = np.ones( (9,9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(imgIn, -1, kernel)
    return(custom)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="path to video",)
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args["video"])
while True:
    (grabbed,frame) = camera.read()
    if not grabbed:
        break
    
    frame = cv2.resize(frame,(450,450))
    M = np.ones(frame.shape,dtype = "uint8")*50
    added = cv2.add(frame,M)
    
    img = cv2.cvtColor(added,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img,(25,25),0)
    img = cv2.medianBlur(img,7)
    for i in range(2):
        img = adjust_sharpness(img)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    edged = cv2.dilate(img,None,iterations = 4)
    """
    edged = cv2.erode(edged,None,iterations = 4)
    #edged = adjust_sharpness(edged)
    thresh = cv2.adaptiveThreshold(edged,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    """
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0 :
        original = frame.copy()
        c = max(cnts, key = cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/ M["m00"]),int(M["m01"] / M["m00"]))
        if radius > 1:
            cv2.circle(original,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(original,center,5,(0,0,255),-1)
            frames = original
    cv2.imshow("Added",frames)
    cv2.waitKey(1)