import cv2 as cv 
import mediapipe as mp
import time

class handDedector():
    def __init__(self,mode=False,maxHands=2,detectionsCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands = maxHands
        self.detectionCon=detectionsCon
        self.trackCon=trackCon
        
        self.mpHands =mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw= mp.solutions.drawing_utils

    def findHands(self,frame):
        imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
               self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def fingPosition(self,frame,handNo=0,draw=True):
        lmList=[]
        if self.result.multi_hand_landmarks:
            myHand=self.result.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm )
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                lmList.append([id,cx,cy])

                if draw:
                    cv.circle(frame,(cx,cy),7,(255,0,0),cv.FILLED)
        return lmList

def main():
    
    pTime=0
    cTime=0

    cap= cv.VideoCapture(0)
    detector=handDedector()
    

    while True:
        success , frame = cap.read()
        frame = detector.findHands(frame)
        lmList=detector.fingPosition(frame)
        if len(lmList)!=0:
            print(lmList[8])
        cTime=time.time()
        fps=1/(cTime-pTime) 
        pTime=cTime
        

        cv.putText(frame, str(int((fps))), (10,70) ,cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        
        cv.imshow('frame',frame)
        cv.waitKey(1)

if __name__ == "__main__":
    main()