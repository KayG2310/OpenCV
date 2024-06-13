import cv2
import mediapipe as mp
import time

class pose_detector():
    def __init__(self, mode=False,upper_body=False,smooth=True, min_detection_conf=0.5,
                 min_tracking_conf = 0.5, model_complexity = 1):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.model_complexity = model_complexity
        

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upper_body,self.smooth,self.min_detection_conf,
                                     self.min_tracking_conf,self.model_complexity)
        self.mpDraw = mp.solutions.drawing_utils
        # Pose file has static image mode set to False, upper body only False, smooth landmarks true
        # and two other detection and tracking confidence set to 0.5
        
    def find_pose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # now we draw our landmarks
        # we have various landmarks here, we can centre on any one
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img       
                
     
    def getposition(self,img,draw=True):
        if self.results.pose_landmarks:
            lmlist = []
            for ids,lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = lm.x*width, lm.y*height 
                # to get integer landmarks
                if id==10:
                    lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx,cy), 35, (123,200,32), cv2.FILLED)
                    
        return lmlist  
    
    

def main():
    capture = cv2.VideoCapture('1.mp4')
    previous_time = 0
    detector = pose_detector()
    while True:
        success,img = capture.read()
        img = detector.find_pose(img)
        lmlist = detector.getposition(img)
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (70,50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3,
                color=(230,12,198), thickness=2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
if __name__ == "__main__":
    main()
    