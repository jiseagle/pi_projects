# -*- coding: utf-8 -*-
#import necessary packages
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

time.sleep(2)

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array
    
    cv2.imshow('Frame', image)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)
    
    if key == ord('q'):
        break


camera.capture(rawCapture, format='bgr')
img= rawCapture.array

cv2.imshow('image', img)
cv2.waitKey(0)                 
cv2.destroyAllWindows()
