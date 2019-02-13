# Author: Paul Mathai
# Co-Author: Rahul Raghunadhan, Akash Smaran, Pranjal Srivastava
# Version: 1.2
# Features: Hough Circles
#           Multi-Camera Support(Stereo)
#           Live Radius Calibration    
#           
import argparse
import cv2
import numpy as np
from skimage import img_as_bool
from skimage import img_as_ubyte
from skimage.morphology import skeletonize
import curses
from time import gmtime, strftime
import time


def adjust_sharpness(imgIn):
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0
    boxFilter = np.ones((9, 9), np.float32) / 81.0
    kernel = kernel - boxFilter
    custom = cv2.filter2D(imgIn, -1, kernel)
    return custom

################################

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
def main(stdscr):
    # Recieveing stream and parsing to opencv
    eyer = cv2.VideoCapture('http://192.168.43.136:8000/eyer.mjpg')
    eyel = cv2.VideoCapture('http://192.168.43.86:8080/eyel.mjpg')
    """
    # Argument Video Feeding
    eyel = cv2.VideoCapture(argsl["lvideo"])
    eyer = cv2.VideoCapture(argsl["rvideo"])
    # Direct video linking
    eyel = cv2.VideoCapture("PI-Streaming/pi_cam_stream.py")
    eyer = cv2.VideoCapture("PI-Streaming/pi_cam_stream.py")
    """
    #eyel = cv2.VideoCapture(0)
    #eyer = eyel
    IMG_HEIGHT = 180
    IMG_WIDTH = 320
    # Angle of the Camera Mount
    ANGLE = 50
    # Copensation for Warp
    LEFT_COMP = (ANGLE/45)
    RIGHT_COMP = 1-LEFT_COMP

    showtime = time.asctime(time.localtime(time.time()))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outl = cv2.VideoWriter(showtime+'_l.mp4', fourcc, 20.0, (640,480))
    outr = cv2.VideoWriter(showtime+'_r.mp4', fourcc, 20.0, (640,480))
    
    arr = [17,61]
    active = 0
    stdscr.nodelay(1)
    a = 0
    
    while eyel and eyer is not None:
        a = a + 1
        (grabbedl, framel) = eyel.read()
        (grabbedr, framer) = eyer.read()
        outl.write(framel)
        outr.write(framer)

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
        #a = get()

        c = stdscr.getch()
        if c != -1:
            # print numeric value
            #stdscr.addstr(str(c) + ' ')
            if(c == 258):
                arr[active] -= 1
            elif(c==259):
                arr[active] += 1
            elif(c == 260):
                active = 0
            elif(c == 261):
                active = 1
            stdscr.refresh()
            # return curser to start position
            stdscr.move(0, 0)

        circlesl = cv2.HoughCircles(imgl, cv2.HOUGH_GRADIENT, 1, minDist=1200000, param1=50, param2=20, minRadius=arr[0],
                                    maxRadius=arr[1])
        circlesr = cv2.HoughCircles(imgr, cv2.HOUGH_GRADIENT, 1, minDist=1200000, param1=50, param2=20, minRadius=arr[0],
                                    maxRadius=arr[1])

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
                cv2.putText(addedl, 'max_radius:' + str(arr[1]), (20, 80), font, 0.6, (255, 255, 255), 1)
                cv2.putText(addedl, 'min_radius:' + str(arr[0]), (20, 100), font, 0.6, (255, 255, 255), 1)
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

if __name__ == '__main__':
    curses.wrapper(main)