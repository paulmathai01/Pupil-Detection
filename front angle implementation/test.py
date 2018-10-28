import numpy as np
import cv2

cap = cv2.VideoCapture('/Desktop/sample.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    frame = cap.read()
    frame = cv2.resize(frame,(480,270))
    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()