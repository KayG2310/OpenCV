import cv2
import mediapipe as mp
import time
capture = cv2.VideoCapture(0)
# initializing camera, 0 is the default camera. Camera index can change if external cameras are added


mpHands = mp.solutions.hands
hands = mpHands.Hands()
#This initializes the MediaPipe Hands solution.
#creates an instance of the Hands class. This object will be used to detect hands in the video frames.
# static_image_mode keeps it slow so it is by default set to False
# minimum tracking and detection confidence is 50%, if it goes lower , the process is done again


mpdraw = mp.solutions.drawing_utils
#  initializes the drawing utilities for drawing the hand landmarks on the image.

previous_time = 0
current_time = 0
# this computes the time difference between frames which is used to calculate FPS


while True:
    # infinite loop to capture video endlessly
    
    success, img = capture.read()
    #boolean to check if the frame was read properly, img is the frame itself
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # self understood
    # converting the image to RGB
    
    results = hands.process(imgRGB) #processes RGB images to detect landmarks (21)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks: #checks if landmarks were detected
        for handslm in results.multi_hand_landmarks:
            for id,lm in enumerate(handslm.landmark):
                # print(id,lm)
                # each hand image will have its landmark listed
                height, width, channels = img.shape
                xcoord, ycoord = int(lm.x*width), int(lm.y*height)
                # print(id, xcoord,ycoord) for all 21 landmarks
                if id==15:
                    cv2.circle(img,(xcoord, ycoord), 25, (255,34,255), cv2.FILLED)
            mpdraw.draw_landmarks(img, handslm, mpHands.HAND_CONNECTIONS)
            # not displaying on rgb image
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                color=(31.4,78.4,47.1), thickness=3,)

    # 10,70 is position
    cv2.imshow("Image", img)
    cv2.waitKey(1) #to keep the window responsive, it waits for 1 millisecond for a key press event i.e detection of a key being presses
