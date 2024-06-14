import cv2
import numpy as np
import time
import mediapipe as mp
from math import hypot as hy
# import pycaw as pc
# not mac os compatible
import subprocess

'''
# volume controls
from ctypes import POINTER,cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute() not needed
# volume.GetMasterVolumeLevel() not needed
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-50.0, None)
min_volume = volRange[0]
max_volume = volRange[1]

#
'''

min_volume = 0
max_volume = 100

# camera parameters go here
cam_width, cam_height = 500, 500
#

# checking if the webcam is working
capture = cv2.VideoCapture(0)
# camera at 0th index works #  ## # # #  ##
capture.set(3, cam_width)
capture.set(4, cam_height)
# #  ## # # #  ##  # # # # #  #  ## # # #  #

# setting up fps  #  ## # # #  ##  # # # # #
previous_time = 0
# # #   # # # # #  # # #  ## # # #  ##  # # #

# setting up hands #  #  # # ## # # # #  # #
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.95, min_tracking_confidence=0.95)
# # #  # #  ## # #  # # ## #  # # #  # # # # #

mpdraw = mp.solutions.drawing_utils


def findhands(img, draw):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handslm in results.multi_hand_landmarks:
            if draw:
                mpdraw.draw_landmarks(img, handslm,
                                      mpHands.HAND_CONNECTIONS)
    return img, results


def position(img, results, handno=0, draw=True):
    landmark_list = []
    if results.multi_hand_landmarks:
        myhand = results.multi_hand_landmarks[handno]
        for id, lm in enumerate(myhand.landmark):
            height, width, channels = img.shape
            xcoord, ycoord = int(lm.x * width), int(lm.y * height)
            landmark_list.append([id, xcoord, ycoord])
            if draw:
                if id==4:
                    cv2.circle(img, (xcoord, ycoord), 18, (255, 255,0),
                               cv2.FILLED)
                if id==8:
                    cv2.circle(img, (xcoord, ycoord), 18, (255, 255, 0),
                               cv2.FILLED)

    return landmark_list


while True:
    success, img = capture.read()
    img = findhands(img, False)[0]
    # at 0th index we have the image
    results = findhands(img,False)[1]
    lmList = position(img, results, draw=True)
    # keeping draw = True will highlight the thumb tip and index fingertip
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        # printing the coordinates of thumb tip and index fingertip
        # this list contains id, x coordinate and y coordinate of the hand and a total of 21 values
        # extracting the coordinates that we need
        x_index, y_index = lmList[8][1], lmList[8][2]
        x_thumb, y_thumb = lmList[4][1], lmList[4][2]
        # to increase or decrease the volume depends on the distance between these coordinates
        # let's depict it with a line

        cv2.line(img, (x_index,y_index),(x_thumb,y_thumb), (255,255,0), 3)
        x_centre, y_centre = (x_thumb+x_index)/2, (y_index+y_thumb)/2
        # mid point of the connecting line

        length_of_line = hy((x_index-x_thumb), (y_index-y_thumb))
        # print(length_of_line)
        vol = np.interp(length_of_line,[50,250], [min_volume,max_volume])
        # volume.SetMasterVolumeLevel(vol,None)
        # Setting volume directly
        if 0 <= vol <= 100:
            volume_level = vol / 10
            script = f"set volume output volume {volume_level * 10}"
            subprocess.run(["osascript", "-e", script])







    current_time = time.time()
    fps = str(int(1 / (current_time - previous_time)))
    previous_time = current_time

    cv2.putText(img, "fps: " + fps, (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (76, 255, 136), 1)
    cv2.imshow("Testing window", img)
    cv2.waitKey(1)
    # adding a delay of 1 millisecond

