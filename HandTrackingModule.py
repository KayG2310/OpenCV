import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        # the same parameters as the mp.solutions.hands file
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mpHands = mp.solutions.hands
        # creating  a hands class
        self.hands = self.mpHands.Hands(self.mode, self.max_num_hands, self.detection_confidence,
                                        self.tracking_confidence)
        # static_image_mode keeps it slow so it is by default set to False
        # minimum tracking and detection confidence is 50%, if it goes lower , the process is done again
        self.mpdraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handslm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handslm,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handno=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                height, width, channels = img.shape
                xcoord, ycoord = int(lm.x * width), int(lm.y * height)
                landmark_list.append([id, xcoord, ycoord])
                if draw:
                    cv2.circle(img, (xcoord, ycoord), 18, (255, 34, 255),
                               cv2.FILLED)

        return landmark_list


def main():
    previous_time = 0
    current_time = 0
    capture = cv2.VideoCapture(0)
    detector = HandDetector()
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


if __name__ == "__main__":
    main()
