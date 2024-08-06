import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)
prevtime = 0

mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(max_num_faces=3)
drawspec = mpdraw.DrawingSpec(thickness=1, circle_radius=1)


while True:
    success,img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img, face,mpfacemesh.FACEMESH_FACE_OVAL,
                                  drawspec,drawspec)
            for id,lm in enumerate(face.landmark):
                print(lm)
                h_,w_,c_ = img.shape
                xco,yco = int(lm.x*w_), int(lm.y*h_)


    curtime = time.time()
    fps = 1/(curtime-prevtime)
    prevtime = curtime

    cv2.putText(img, "frame rate: "+str(int(fps)), (10,70),cv2.FONT_ITALIC,
                1,(150,78,90),2)
    cv2.imshow("mesh",img)
    cv2.waitKey(1)
