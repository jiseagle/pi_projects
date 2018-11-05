# -*- coding:utf-8 -*-
# Author:Norman Chen
# Subject: Face Recognization
# Version:0.0.0
# Date:2018-11-04
# import necesarry packages
import face_recognition
import argparse
import pickle
import cv2

# construct arguments parser and parse arguments
ap =argparse.ArgumentParser()
ap.add_argument('-e', '--encodings', required=True, help='path to serialized db of facial encodings')
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-d', '--detection-method', type=str, default='hog',
                help='face detection model to use: either "hog" or "CNN"')
args=vars(ap.parse_args())

# load the known faces and embeddings
print('[INFO] loading encodings...')
data = pickle.loads(open(args['encodings'], 'rb').read())

# load the input image and cover it from BGR to RGB
image = cv2.imread(args['image'])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)--coordinates of the bouding boxes corresponding to each face in the input image.
# then compute tje facial embeddings for each face.
print('[INFO] recognizing face ....')
boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
encodings = face_recognition.face_encodings(rgb, boxes)

# init name list for each face detected
names =[]

# loop oever the facial embeddings
for encoding in encodings:
    # to match each face in the input image to our known encodings
    matches = face_recognition.compare_faces(data['encodings'], encoding)
    name='Unknown'
    
    # check if we find a matched name
    if True in matches:
        # find the index of all matched faces, then init a directionary
        # to count total number of times each face was matched
        matchesIndex = [i for (i, b) in enumerate(matches) if b]
        counts={}
        
        # loop over the matched indexes and maintain a count for
        # each recongnized face
        for i in matchesIndex:
            name = data['names'][i]
            counts[name]=counts.get(name, 0) + 1
        
        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie python will
        # select first entry in the dictionary)
        name = max(counts, key=counts.get)
        
    # update the list of name
    names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw the predicted name of face on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2)
    y = top -15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# show the result
cv2.imshow('image', image)
print('[INFO] face recognition complete.....')
cv2.waitKey(0)
cv2.destroyAllWindows()
    
        
