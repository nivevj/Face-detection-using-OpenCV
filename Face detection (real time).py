# -*- coding: utf-8 -*-
"""
Face detection using realtime video
@author: Nivedha
"""
import cv2

#load the cascade
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture video from webcame
vid=cv2.VideoCapture(0)

#performing face detection for each frame
while(True):
    #reading frame
    __, img=vid.read()
    #converting to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect faces
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    #draw rectangle around each face
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #display
    cv2.imshow('img',img)
    #to stop
    x=cv2.waitKey(30) & 0xff
    if x==27:
        break

#release videocapture object
vid.release()


