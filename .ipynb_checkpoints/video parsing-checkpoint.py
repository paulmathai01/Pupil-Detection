import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to video", )
args = vars(ap.parse_args())

print (args["video"])
vidcap = cv2.VideoCapture(args["video"])
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
