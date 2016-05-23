import cv2
import numpy as np
import socket
import time
import multiprocessing
import mutex
print "hello world"

cap = cv2.VideoCapture(0)

while (True):
    frame = cap.read()
    cv2.imshow('Image Name', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()