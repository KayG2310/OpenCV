import numpy as np

# import autopy
# might work on windows but as of now is not working on MacOS

import pyautogui
import cv2
import numpy
import mediapipe as mp
import time
from math import hypot as hy
# modules are now installed

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
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
                if id == 8:
                    cv2.circle(img, (xcoord, ycoord), 18, (255, 0, 255),
                               cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (xcoord, ycoord), 18, (255, 0, 255),
                               cv2.FILLED)

    return landmark_list


screen_width, screen_height = pyautogui.size()
# successful
print(screen_width, screen_height)
# setting our screen width and size to be mapped with window screen size

# our hand tracking module is also imported like that

# capturing the video now
capture = cv2.VideoCapture(0)
frameR = 100
# this is the frame reduction
# due to handling y coordinates properly
# setting the capture dimensions
capture.set(3, 700)
capture.set(4, 700)


smoothening = 5
# this is to reduce the shakiness of the pointer
# like drawing we will also need the previous x and y coordinates
previous_x, previous_y = 0, 0
current_x, current_y = 0, 0
previous_time = 0
while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    # we now need the tips of index and middle finger, calling our pseudo classes

    img = findhands(img, draw=True)[0]
    res = findhands(img, draw=False)[1]
    lmList = position(img, res, draw=True)

    # checking which fingers are up based on which we'll define our rules
    # similar to our drawing mode, index finger is moving
    # both index and middle are selection mode
    # convert coordinates (our window should be mapped to the entire screen size)
    # we smoothen those values of the new coordinates
    # distance between the two fingers, if it's short, click the mouse

    # adding frame rate
    current_time = time.time()
    fps = int(1/(current_time - previous_time))
    previous_time = current_time


    cv2.putText(img, str(fps), (20,50), cv2.FONT_HERSHEY_PLAIN, 2,
                (100,150,200), 2)

    if len(lmList) != 0:
        x_index, y_index = lmList[8][1], lmList[8][2]
        x_middle, y_middle = lmList[12][1], lmList[12][2]

        ups = [0,0]
        if y_index < lmList[6][2]:
            ups[0] = 1
        if y_middle < lmList[10][2]:
            ups[1] = 1

        # now our actions depend on the 0s and 1s of the ups
        cv2.rectangle(img, (frameR, frameR), (1200, 680), (0, 200, 0), 2)
        # the purpose of the box is that when you are at the top of the box, you're at the top
        # of the screen,
        if ups[0] == 1 and ups[1] == 0:
            # meaning only the index finger is up we are in our selection mode
            x3 = np.interp(x_index, (frameR, 1200), (0, screen_width))
            y3 = np.interp(y_index, (frameR, 680), (0, screen_height))

            # adding the smoothening code here
            current_x = previous_x + (x3-previous_x)/smoothening
            current_y = previous_y + (y3-previous_y)/smoothening



            pyautogui.moveTo(current_x, current_y)
            previous_x, previous_y = current_x, current_y


        # now moving to the selection mode
        if ups[0] == 1 and ups[1] == 1:
            length_of_line = hy((x_index - x_middle), (y_index - y_middle))
            # length less than 40 will be detected as a click
            if length_of_line < 53 :
                cv2.circle(img, (x_index, y_index), 20, (0,255,0),
                           cv2.FILLED)
                pyautogui.click()
                # it will click, the issue here is that the mouse is very shaky





    cv2.imshow("mouse", img)
    cv2.waitKey(1)
