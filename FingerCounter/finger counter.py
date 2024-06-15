import mediapipe as mp
import cv2
import numpy as np
import time
import os

capture = cv2.VideoCapture(0)
previous_time = 0


while True:
    success,img = capture.read()
    cv2.imshow(img)