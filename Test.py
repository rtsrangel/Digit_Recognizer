import cv2
import numpy as np
import socket
import time
import multiprocessing
import mutex
#import sklearn

print "hello world"
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(15, -3.0)

while (True):
    ret, frame = cap.read()
    cv2.imshow('Image Name', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
