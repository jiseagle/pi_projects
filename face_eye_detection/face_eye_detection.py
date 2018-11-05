# -*- coding:utf-8 -*-
# Author:Norman Chen
# Subject: Face and Eye Detection
# Version:0.0.0
# Date:2018-11-05
# import necesarry packages
import os
import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

# load OpenCV's Haar Cascade for face detection from disk
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector.load('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eyes_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
eyes_detector.load('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# define the recording video and codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('frame.avi', fourcc, 15, (960,720))

# initialize the video stream, allow the camera sensor to warm up
print('[INFO] starting video streaming')
camera = PiCamera()
camera.resolution=(960,720)
camera.framerate=15.0
rawCapture = PiRGBArray(camera, size=(960,720))
time.sleep(2)

# loop over frames from video stream
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    img = cv2.flip(frame.array,1)
        
    # grab the frame from video stream, and clone it.
    # rsize the frame to apply fast detection faster
    #frame = vs.read()
    orig = img.copy()
    #frame = imutils.resize(frame, width=400)
        
    # change BGR faces in grayscale
    frameGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face in grayscale
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
            
    # if 'q' is pressed, then break from this loop
    if key == ord('q'):
        break

# print how many faces saved
print('[INFO] {} video stored in this time'.format('frame.avi'))
print('[INFO] End video recording...')

cv2.destroyAllWindows()
