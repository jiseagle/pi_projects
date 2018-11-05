# -*- coding:utf-8 -*-
# Author:Norman Chen
# Subject: Search Google Image API
# Version:0.0.0
# Date:2018-11-04
# import necesarry packages
from imutils import paths
from requests import exceptions
import argparse
import requests
import cv2
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-u', '--urls', required=True, help='path of filecontaining image URL')
ap.add_argument('-o', '--output', required=True, help='path to output directory of images')
args = vars(ap.parse_args())

# grab the list from URL file, and intialize total number of images downloaded
rows = open(args['urls']).read().strip().split('\n')
total = 0

EXCEPTIONS = set([exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError,
                  exceptions.ConnectTimeout, exceptions.Timeout, FileNotFoundError])
# loop the url
for url in rows:
    print(url)
    try:
        # download the images
        r = requests.get(url, timeout=60)
        
        # save images
        p = os.path.sep.join([args['output'], "{}.jpg".format(str(total).zfill(8))])
        f = open(p, 'wb')
        f.write(r.content)
        f.close()
        
        # update count for total
        print('[INFO] downloaded ...: {}'.format(p))
        total += 1
        
    except Exception as e:
        if type(e) in EXCEPTIONS:
            print('[INFO] Exception happened, error download {} ...'.format(p))

# loop over the image paths we download
for imagePath in paths.list_images(args['output']):
    # init delete as False to determine if the image shall be deleted or not.
    delete = False
    
    # load the image
    try:
        image = cv2.imread(imagePath)
        
        # if image is "None", the we woll delete it
        if image is None:
            delete = True
    
    # if OpenCV cannot load image, then delete image
    except:
        print('[INFO] {} Image is None ..'.format(imagePath))
        delete = True
    
    # if delete is True, then delete image
    if delete == True:
        print('[INFO] deleting {}'.format(imagePath))
        os.remove(imagePath)
