import argparse

import cv2
import numpy as np
from imutils.video import FPS
from skimage import img_as_bool
from skimage import img_as_ubyte
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

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

fgbg = cv2.createBackgroundSubtractorMOG2()

vs = cv2.VideoCapture(args["video"])
a = 0
while True:
    a = a + 1
    (grabbed, frame) = vs.read()

    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    M = np.ones(frame.shape, dtype="uint8") * 50
    added = cv2.add(frame, M)
    added = cv2.cvtColor(added, cv2.COLOR_BGR2GRAY)
    print("[INFO] Converted from RGB to GRAY for interation=", a)
    img = cv2.equalizeHist(added)
    img = cv2.medianBlur(img, 1)
    img = cv2.GaussianBlur(img, (17, 17), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 47, 6)
    img = cv2.GaussianBlur(img, (17, 17), 0)

    rows, cols = img.shape
    edges = canny(img, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    print("[INFO] Edges Detected for interation=", a)

    result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=27, max_size=83)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    added[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    # img = img_as_bool(img)
    # img = skeletonize(img)
    # img = img_as_ubyte(img)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=1200000, param1=20, param2=18, minRadius=27, maxRadius=83)
    """if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if x > (IMG_WIDTH * 0.3) and (IMG_HEIGHT * 0) < y < (IMG_HEIGHT * 1):
                cv2.circle(added, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(added, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), 1)
                print("[INFO] Computing for (", x, ",", y, ",", r, ")")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(added, 'x:' + str(x), (20, 20), font, 0.6, (0, 0, 255), 1)
            cv2.putText(added, 'y:' + str(y), (20, 40), font, 0.6, (0, 0, 255), 1)
            cv2.putText(added, 'r:' + str(r), (20, 60), font, 0.6, (0, 0, 255), 1)
    # img = cv2.addWeighted(added, 1, img, 1, 1)"""
    cv2.imshow("Added", edges)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
