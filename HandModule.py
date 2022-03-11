from cv2 import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max=2, detect=0.5, track=0.5):
        self.mode=mode
        self.max=max
        self.detect=detect
        self.track=track
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max,self.detect,self.track)
        self.draw = mp.solutions.drawing_utils


    def findHands(self, img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.rezHand = self.hands.process(imgRGB)
        if self.rezHand.multi_hand_landmarks:
            for handLms in self.rezHand.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handN=0, draw=True):
        listaLM=[]
        if self.rezHand.multi_hand_landmarks:
            myHand=self.rezHand.multi_hand_landmarks[handN]
            for id, lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                listaLM.append([id,cx,cy])
        return listaLM








def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    det=handDetector()
    while True:
        success, img = cap.read()
        img = det.findHands(img)
        lmList=det.findPosition(img)
        if len(lmList)!=0:
            print(lmList[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
