import argparse

import cv2
import numpy as np
from imutils.video import FPS

IMG_HEIGHT = 270
IMG_WIDTH = 480


def adjust_sharpness(imgIn):
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(imgIn, -1, kernel)
    return custom


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to video", )
args = vars(ap.parse_args())

fps = FPS().start()

# fgbg = cv2.createBackgroundSubtractorMOG2()

vs = cv2.VideoCapture(args["video"])
a = 0
while True:
    a = a + 1
    (grabbed, frame) = vs.read()

    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    # frame = frame[0:170, 100:320]
    # print("[INFO] Resizing")
    M = np.ones(frame.shape, dtype="uint8") * 50
    added = cv2.add(frame, M)
    img = cv2.cvtColor(added, cv2.COLOR_BGR2GRAY)
    print("[INFO] Converted from RGB to GRAY for interation=", a)
    img = cv2.equalizeHist(img)
    img2 = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.dilate(img, None, iterations = 5)
    # img = cv2.erode(img, None, iterations=5)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 121, 3)
    # fgmask = fgbg.apply(img)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, minDist=200, param1=100, param2=4
                               , minRadius=25, maxRadius=39)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if (IMG_WIDTH * 0.4) < x < (IMG_WIDTH * 0.6) and y > (IMG_HEIGHT * 0.3) and y < (IMG_HEIGHT * 0.75):
                cv2.circle(frame, (x, y), r, (255, 255, 255), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
            print("[INFO] Computing for (", x, ",", y, ",", r, ")")
            font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'x:' + str(x), (20, 20), font, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, 'y:' + str(y), (20, 40), font, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, 'r:' + str(r), (20, 60), font, 0.6, (0, 0, 255), 1)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
