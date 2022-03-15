from cv2 import cv2
import HandModule as hm
import pyautogui


###############################
# Mouse control via hand gestures
# Index finger to move the mouse, thumb to click
###############################


cap=cv2.VideoCapture(0)
det=hm.handDetector(detect=0.7)
klik=0
xold,yold=0,0

while True:
    succ,img=cap.read()
    img = det.findHands(img, False)
    lmList = det.findPosition(img)
    h, w, ch = img.shape
    # border 54, 90
    # 532, 300
    # 640x480
    cv2.rectangle(img,(54,90),(586,390),(255,0,255),3)


    if len(lmList) != 0:

        thumbx, thumby = lmList[4][1], lmList[4][2]
        knucklex, knuckley = lmList[3][1], lmList[3][2]
        pointerx, pointery = lmList[8][1], lmList[8][2]
        cv2.circle(img, (pointerx,pointery),10,(255,0,255),cv2.FILLED)

        mw,mh=pyautogui.size()
        if pointerx>53 and pointerx<587 and pointery>89 and pointery<390:

            mousex=int((1-(pointerx-54)/532) *mw)
            mousex=mousex//5
            mousex=mousex*5+2
            mousey=int((pointery-90)/300 *mh)
            mousey=mousey//5
            mousey=mousey*5+2
            pyautogui.moveTo(mousex,mousey)


        if thumbx<knucklex:
            pyautogui.click()




    cv2.imshow("LiveFeed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()