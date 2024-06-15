import mediapipe as mp
import cv2
import numpy as np
import time
import os
import mediapipe as mp

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


cam_width, cam_height = 500, 500
capture = cv2.VideoCapture(0)
capture.set(3, cam_width)
capture.set(4, cam_height)

folder = "hands_images"
list_of_fings = os.listdir(folder)
# print(list_of_fings)
list_of_fings.sort(key=lambda f: int(os.path.splitext(f)[0]))
overlist = []
for image_path in list_of_fings:
    image = cv2.imread(f'{folder}/{image_path}')
    image = cv2.resize(image,(150,150))
    overlist.append(image)

previous_time = 0

# adding hand tracking modules above

tips_id = [4,8,12,16,20]
# tips of all fingers starting from thumb to pinky finger

while True:
    tips = []
    success,img = capture.read()

    img = findhands(img, True)[0]
    results = findhands(img,True)[1]
    lmList = position(img, results, draw=False)
    if len(lmList)!=0:
        # the finger is raised or not depends on the tip distance from just above the palm
        # e.g. checking it for one finger up
        '''
                if lmList[8][2] < lmList[6][2]:
            # above that point
            # implies the finger is lowered
            print("index finger open")
            # one of the way to check for the tips, we need to do it for 5 fingers
        '''

        # thumb case
        if lmList[4][1] > lmList[3][1]:

            # implies that finger is open
            tips.append(1)
            # end of this for loop gives us the number of open fingers

        else:
            tips.append(0)

        for ids in tips_id:
            if ids ==4:
                continue
            if lmList[ids][2] < lmList[ids-2][2]:

                # implies that finger is open
                tips.append(1)
                # end of this for loop gives us the number of open fingers

            else :
                tips.append(0)

        # print(tips)
        # one drawback - even if the fist is closed, it counts it as an open thumb
        # handling the thumb separately,
        total_fingers = tips.count(1)
        img[0:150, 0:150] = overlist[total_fingers]
        cv2.rectangle(img,(5, 200), (130,300), (0,200,0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (50, 300), cv2.FONT_HERSHEY_PLAIN,
                    4,(0,0,0),2)


        # print(total_fingers) this works fine

    current_time = time.time()
    fps = str(int(1/(current_time-previous_time)))
    previous_time = current_time
    cv2.putText(img, "fps: "+fps, (200,70), cv2.FONT_HERSHEY_PLAIN, 1,
                (50,100,150),1)

    cv2.imshow("Fingers",img)
    cv2.waitKey(1)


