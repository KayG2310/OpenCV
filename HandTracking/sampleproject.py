import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
previous_time = 0
current_time = 0
capture = cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = capture.read()
    img = detector.findhands(img)
    our_list = detector.findposition(img)
    if len(our_list) != 0:
        print(our_list[4])
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3,
                (255, 0, 255), 3)

    # 10,70 is position
    cv2.imshow("Image", img)
    cv2.waitKey(1)
