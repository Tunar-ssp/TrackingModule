import cv2 as cv
import time
import mediapipe as mp


class faceDetector():
    def __init__(self, minCon=0.5, modelSelection=0):
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            min_detection_confidence=minCon,
            model_selection=modelSelection
        )

    def findFace(self,frame,drawFace=True,writeCon=True):
        imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results=self.faceDetection.process(imgRGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                boxCor=detection.location_data.relative_bounding_box
                ih,iw,ic=frame.shape
                box= int(boxCor.xmin*iw) , int(boxCor.ymin*ih),\
                    int(boxCor.width *iw), int(boxCor.height*ih)
                if drawFace:
                    self.drawFace(frame,box)
                if writeCon:
                    self.writeCon(frame,box,detection.score)
            

                
    def drawFace(self,frame,box):
        cv.rectangle(frame,box,(255,0,255),3)
    def writeCon(self,frame,box,score):
        cv.putText(frame,
                f'Detection score:{int(score[0]*100)}%',
                (box[0],box[1]-20),
                cv.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
class handDetector():
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

    def findHands(self,frame,draw=True):
        imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if draw:
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
class poseDetector():
    

    def __init__(self,mode=False,modelComplexity=1,upBody=False,smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon 
        self.trackCon=trackCon
        self.modelComplexity=modelComplexity

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(
               static_image_mode=self.mode,
               model_complexity=self.modelComplexity,
               smooth_landmarks=self.smooth,
               enable_segmentation=self.upBody,
               smooth_segmentation=True,
               min_detection_confidence=self.detectionCon,
               min_tracking_confidence=self.trackCon
            )
                                                        

    def findPose(self,frame,draw=True):
        imgRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return frame
    

    def getPositions(self,frame,draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm  in enumerate(self.results.pose_landmarks.landmark):
                h , w , c=frame.shape
                # print(id,lm)
                cx,cy= int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(frame,(cx,cy),10,(255,0,255),cv.FILLED)
            
        return lmList
    
def main():
    handTrack=handDetector()
    poseTrack=poseDetector()
    faceTrack=faceDetector()
    video=cv.VideoCapture(0)
    pTime=0
    while True:
        success, frame= video.read()
        # Test
        handTrack.findHands(frame)
        poseTrack.findPose(frame)
        faceTrack.findFace(frame)
        # 


        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv.putText(frame, f'Fps: {int(fps)}', (10,30), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        cv.imshow('Video',frame)
        cv.waitKey(1)








if __name__=="__main__":
    main()
