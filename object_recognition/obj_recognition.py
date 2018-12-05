# -*- coding:utf-8 -*-
# Author:Norman Chen
# Subject: Object Recognization
# Version:0.0.0
# Date:2018-11-04
# import necesarry packages
import numpy as np
import argparse
import time
import cv2
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from imutils.video import FPS

ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True, help ='path to input image')
ap.add_argument('-p', '--prototxt', required=True, help='path o Caffe')
ap.add_argument('-m', '--model', required=True, help ='path to pretrained model')
ap.add_argument('-l', '--lables', required=True, help ='path to ImageNet labels')
args = vars(ap.parse_args())

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('object.avi', fourcc, 15, (640,480))
#init Pi Camera
#camera = PiCamera()
#camera.resolution=(320,240)
#camera.framerate=30.0
#rawCapture = PiRGBArray(camera, size=(320,240))
#time.sleep(2)
fps=FPS().start()

rows = open(args['lables']).read().strip().split('\n')
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
#    image = cv2.flip(frame.array,1)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while(cap.isOpened()):
    ret, image = cap.read()
    
    blob = cv2.dnn.blobFromImage(image, 1, (244, 244), (104, 117, 123))

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)
    #start = time.time()
    preds = net.forward()
    #end = time.time()
    #print("[INFO] classification took {:.5} seconds".format(end - start))

    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    idxs = np.argsort(preds[0])[::-1][:5]

    # loop over the top-5 predictions and display them
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
            cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # display the predicted label + associated probability to the
        # console
        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))

    
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
    
    cv2.imshow("Image", image)
    out.write(image)
    fps.update()
    fps.stop()
    print('[INFO] fps is --> {:.2f}'.format(fps.fps()))


cv2.destroyAllWindows()
