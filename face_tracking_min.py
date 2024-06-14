import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)
previous_time = 0

mpFacedetect = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceD = mpFacedetect.FaceDetection()


while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceD.process(imgRGB)
    # print result just gives the solution class
    if results.detections:
        for detection in results.detections:
            # print(detection)
            # print(detection.location_data.relative_bounding_box)
            # gives normalized pixel values
            # mpDraw.draw_detection(img,detection)
            bounding_box = detection.location_data.relative_bounding_box
            h, w, chann = img.shape
            bb = int(bounding_box.xmin*w), int(bounding_box.ymin*h), \
                    int(bounding_box.width*w), int(bounding_box.height*h)

            cv2.rectangle(img,bb,(255,29,92),2)
            cv2.putText(img, "score: "+str(detection.score[0]),(70,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (200,56,127),2)

    current_time = time.time()
    fps = int(1/(current_time-previous_time))
    previous_time = current_time
    cv2.putText(img, str(fps),(10,70), cv2.FONT_HERSHEY_TRIPLEX,3,(23,10,78),5)
    cv2.imshow("Face", img)
    cv2.waitKey(1)