# -*- coding: utf-8 -*-
#import necessary packages
import cv2
#import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution=(640,480)
camera.framerate=20.0
rawCapture = PiRGBArray(camera, size=(640,480))
time.sleep(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = cv2.flip(frame.array,1)
    
    #out.write(image)
    t1 = image[300:400, 200:300]
    image[100:200, 100:200] = t1
    
    cv2.imshow('frame', image)
    rawCapture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break       

cv2.destroyAllWindows()

