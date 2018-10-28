# Author: Paul Mathai
# Version: 1.1
# Features: Hough Circles
#           Multi-Camera Support

import argparse
import cv2
import numpy as np
from skimage import img_as_bool
from skimage import img_as_ubyte
from skimage.morphology import skeletonize

# image resolution to be processed
IMG_HEIGHT = 135
IMG_WIDTH = 240
# Angle of the Camera Mount
ANGLE = 38.7
# Copensation for Warp
LEFT_COMP = (ANGLE/45)
RIGHT_COMP = 1-LEFT_COMP

def adjust_sharpness(imgIn):
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(imgIn, -1, kernel)
    return custom

"""
# Argument parsing Left eye video
apl = argparse.ArgumentParser()
apl.add_argument("-lv", "--lvideo", required=True, help="path to left video", )
argsl = vars(apl.parse_args())

# Argument parsing Right eye video
apr = argparse.ArgumentParser()
apr.add_argument("-rv", "--rvideo", required=False, help="path to right video", )
argsr = vars(apr.parse_args())
"""

# Recieveing stream and parsing to opencv
eyel = cv2.VideoCapture('http://192.168.43.253:8000/eyel.mjpeg')
eyer = cv2.VideoCapture('http://192.168.43.42:8000/eyer.mjpeg')
"""
# Argument Video Feeding
eyel = cv2.VideoCapture(argsl["lvideo"])
eyer = cv2.VideoCapture(argsl["rvideo"])


# Direct video linking
eyel = cv2.VideoCapture("/Users/paulmathai/Desktop/Eye-Tracking-Videos/trial3.mp4")
eyer = cv2.VideoCapture("/Users/paulmathai/Desktop/Eye-Tracking-Videos/asdf.mp4")
"""
a = 0
while eyel and eyer is not None:
    a = a + 1
    (grabbedl, framel) = eyel.read()
    (grabbedr, framer) = eyer.read()

    framel = cv2.resize(framel, (IMG_WIDTH, IMG_HEIGHT))
    framer = cv2.resize(framer, (IMG_WIDTH, IMG_HEIGHT))

    # print("[INFO] Resizing")

    Ml = np.ones(framel.shape, dtype="uint8") * 50
    Mr = np.ones(framer.shape, dtype="uint8") * 50

    addedl = cv2.add(framel, Ml)
    addedr = cv2.add(framer, Mr)

    addedl = cv2.cvtColor(addedl, cv2.COLOR_BGR2GRAY)
    addedr = cv2.cvtColor(addedr, cv2.COLOR_BGR2GRAY)

    print("[INFO] Converted from RGB to GRAY for interation=", a)

    imgl = cv2.equalizeHist(addedl)
    imgr = cv2.equalizeHist(addedr)

    # imgl = cv2.bilateralFilter(imgl, 20, 12, 12, cv2.BORDER_DEFAULT)

    imgl = cv2.medianBlur(imgl, 41)
    imgr = cv2.medianBlur(imgr, 41)

    # imgl = cv2.GaussianBlur(imgl, (17, 17), 3, 3)
    # imgl = adjust_sharpness(imgl)

    imgl = cv2.adaptiveThreshold(imgl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 47, 6)
    imgr = cv2.adaptiveThreshold(imgr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 47, 6)

    imgl = cv2.GaussianBlur(imgl, (5, 5), 0, 0)
    imgr = cv2.GaussianBlur(imgr, (5, 5), 0, 0)
    # rows, cols = imgl.shape

    pts3 = np.float32([[0, 0], [IMG_WIDTH, 0], [0, IMG_HEIGHT], [IMG_WIDTH, IMG_HEIGHT]])
    pts4 = np.float32(
        [[0, 0], [IMG_WIDTH, 0], [(IMG_WIDTH * RIGHT_COMP), IMG_HEIGHT], [(IMG_WIDTH * LEFT_COMP), IMG_HEIGHT]])

    PT = cv2.getPerspectiveTransform(pts3, pts4)

    imgl = cv2.warpPerspective(imgl, PT, (IMG_WIDTH, IMG_HEIGHT))
    imgr = cv2.warpPerspective(imgr, PT, (IMG_WIDTH, IMG_HEIGHT))

    addedl = cv2.warpPerspective(addedl, PT, (IMG_WIDTH, IMG_HEIGHT))
    addedr = cv2.warpPerspective(addedr, PT, (IMG_WIDTH, IMG_HEIGHT))

    # imgl = cv2.bitwise_not(imgl)

    imgl = img_as_bool(imgl)
    imgr = img_as_bool(imgr)

    imgl = skeletonize(imgl)
    imgr = skeletonize(imgr)

    imgl = img_as_ubyte(imgl)
    imgr = img_as_ubyte(imgr)
    circlesl = cv2.HoughCircles(imgl, cv2.HOUGH_GRADIENT, 1, minDist=1200000, param1=50, param2=17, minRadius=11,
                                maxRadius=31)
    circlesr = cv2.HoughCircles(imgr, cv2.HOUGH_GRADIENT, 1, minDist=1200000, param1=50, param2=17, minRadius=11,
                                maxRadius=31)

    if circlesl is not None:
        circlesl = np.round(circlesl[0, :]).astype("int")
        for (x, y, r) in circlesl:
            if x > (IMG_WIDTH * 0.3) and (IMG_HEIGHT * 0) < y < (IMG_HEIGHT * 1):
                cv2.circle(addedl, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(addedl, (x - 5, y - 5), (x + 5, y + 5),
                              (255, 255, 255), 1)
                print("[INFO] Computing for Left (", x, ",", y, ",", r, ")")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(addedl, 'x:' + str(x), (20, 20), font, 0.6, (255, 255, 255), 1)
            cv2.putText(addedl, 'y:' + str(y), (20, 40), font, 0.6, (255, 255, 255), 1)
            cv2.putText(addedl, 'r:' + str(r), (20, 60), font, 0.6, (255, 255, 255), 1)
    if circlesr is not None:
        circlesr = np.round(circlesr[0, :]).astype("int")
        for (x, y, r) in circlesr:
            if x > (IMG_WIDTH * 0) and (IMG_HEIGHT * 0) < y < (IMG_HEIGHT * 1):
                cv2.circle(addedr, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(addedr, (x - 5, y - 5), (x + 5, y + 5),
                              (255, 255, 255), 1)
                print("[INFO] Computing for Right (", x, ",", y, ",", r, ")")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(addedr, 'x:' + str(x), (20, 20), font, 0.6, (255, 255, 255), 1)
            cv2.putText(addedr, 'y:' + str(y), (20, 40), font, 0.6, (255, 255, 255), 1)
            cv2.putText(addedr, 'r:' + str(r), (20, 60), font, 0.6, (255, 255, 255), 1)
    # imgl = cv2.addWeighted(addedl, 1, imgl, 1, 1)
    cv2.imshow("Left", addedl)
    cv2.imshow("Right", addedr)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
eyel.stop()
