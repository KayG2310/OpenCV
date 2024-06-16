import os
import cv2
import mediapipe as mp
import time
import numpy as np

# we have these functions instead of hand module
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.83, min_tracking_confidence=0.83)
mpdraw = mp.solutions.drawing_utils
draw_color = (139, 0, 0)


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
                    cv2.circle(img, (xcoord, ycoord), 12, draw_color,
                               cv2.FILLED)
    return landmark_list


# next step is to import those images
folder = "tools_images"
list_of_images = os.listdir(folder)
list_of_images = [f for f in list_of_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# print(list_of_images)
list_of_images.sort(key=lambda f: int(os.path.splitext(f)[0]))
# sorting the images in custom order

overlist = []
for image_path in list_of_images:
    image = cv2.imread(f'{folder}/{image_path}')
    overlist.append(image)
# print(len(overlist))
header = overlist[0]
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

xprev, yprev = 0, 0
actual_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # reading the capture
    # flipping the image because otherwise it is difficult to keep a record of the drawing for the user
    success, img = capture.read()
    img = cv2.flip(img, 1)

    # using our two functions as modules for providing image and the landmarks list
    img = findhands(img, draw=True)[0]
    results = findhands(img, draw=False)[1]
    lmList = position(img, results, draw=True)

    # we need to define some rules of our canvas, when two fingers are up, it is the selection mode
    if len(lmList) != 0:

        xindex, yindex = lmList[8][1], lmList[8][2]
        xmiddle, ymiddle = lmList[12][1], lmList[12][2]

        # we now have to use the finger counter for these two
        ups = []
        if yindex < lmList[6][2]:
            ups.append(1)
        if ymiddle < lmList[10][2]:
            ups.append(1)
        # print(ups)

        # now depending on the length of ups, our functions will be decided
        fingers = len(ups)
        if fingers >= 1:
            if fingers == 2:
                # print("Selecting tool")
                # diving more into tool selection
                xprev, yprev = None, None
                # this is to avoid those unnecessary straight lines
                if yindex < 125:
                    # less than the header value that is
                    # nested ifs to see which tool we are selecting
                    if 50 < xindex < 250:
                        # blue tool is selected
                        # print("blue \n")
                        header = overlist[1]
                        draw_color = (250, 0, 0)
                    elif 253 < xindex < 450:
                        header = overlist[2]
                        draw_color = (153, 255, 255)
                    elif 453 < xindex < 650:
                        header = overlist[3]
                        draw_color = (52, 66, 227)
                    elif 653 < xindex < 850:
                        header = overlist[4]
                        draw_color = (60, 179, 113)

                    else:
                        header = overlist[5]
                        draw_color = (0, 0, 0)

            else:
                # print("Drawing mode")
                cv2.circle(img, (xindex, yindex), 9, draw_color, 1)
                if xprev is None and yprev is None:
                    xprev, yprev = xindex, yindex
                if draw_color != (0, 0, 0):
                    cv2.line(img, (xprev, yprev), (xindex, yindex), draw_color, 14)
                    cv2.line(actual_canvas, (xprev, yprev), (xindex, yindex), draw_color, 14)

                else:
                    cv2.line(img, (xprev, yprev), (xindex, yindex), draw_color, 100)
                    cv2.line(actual_canvas, (xprev, yprev), (xindex, yindex), draw_color, 100)
                xprev, yprev = xindex, yindex
    else:
        xprev, yprev = None, None
    # meaning there's some actual content
    # rule - in this our main focus is on the tip of index and middle finger
    # when one finger is up(index) drawing should be done
    #
    # setting the header image

    img_gray = cv2.cvtColor(actual_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, actual_canvas)

    img[0:125, 0:1280] = header
    # to have it in same window
    # img = cv2.addWeighted(img,0.5, actual_canvas,0.5,0)
    # this is a good translucent board, opacity is affected, and the brightness of colours

    # to have a separate canvas
    cv2.imshow("paint", img)
    # cv2.imshow("Canvas", actual_canvas)
    # cv2.imshow("inverse", imgInverse)
    # # # # # # # # ##
    cv2.waitKey(1)
