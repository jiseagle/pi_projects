# -*- coding:utf-8 -*-
# Author:Norman Chen
# Subject: Build Face Dataset
# Version:0.0.0
# Date:2018-11-04
# import necesarry packages
import argparse
import os
import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

#construct argment parse and parse the argument
ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--cascade", required = True,
#                help="path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
args = vars(ap.parse_args())

# load OpenCV's Haar Cascade for face detection from disk
#detector = cv2.CascadeClassifier(args["cascade"])
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector.load('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eyes_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
eyes_detector.load('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# define the recording video and codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('frame.avi', fourcc, 15, (960,720))

# initialize the video stream, allow the camera sensor to warm up
# and initialize the total niumber of exmaple faces wriiten to disk
print('[INFO] starting video streaming')
camera = PiCamera()
camera.resolution=(960,720)
#camera.framerate=15.0
rawCapture = PiRGBArray(camera, size=(960,720))
time.sleep(2)
# fix below values for in-room low light mode
camera.iso = 200
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g=camera.awb_gains
camera.awb_mode ='off'
camera.awb_gains=g
total=0

# loop over frames from video stream
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    img = cv2.flip(frame.array,1)
        
    # grab the frame from video stream, and clone it.
    # rsize the frame to apply fast detection faster
    #frame = vs.read()
    orig = img.copy()
    #frame = imutils.resize(frame, width=400)
        
    # detect faces in grayscale
    frameGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    # loop over thr face detection and draw them on the frame
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        roi_gray = frameGray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eyes_detector.detectMultiScale(roi_gray,scaleFactor=2.0, minNeighbors=3, minSize=(5,5))
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)

    # show the output frame
    out.write(img)
    cv2.imshow('Frame', img)
    frame.truncate(0)
    key = cv2.waitKey(1) & 0xFF
            
    # if 'k' is pressed, write the original frame to disk
    # then we can process it for face recognition
    if key == ord('k'):
        p = os.path.sep.join([args["output"],"{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, orig)
        total+=1
            # if 'q' is pressed, then break from this loop
    if key == ord('q'):
        break

# print how many faces saved
print('[INFO] {} face images stored in this time'.format(total))
print('[INFO] cleaning up...')

cv2.destroyAllWindows()
