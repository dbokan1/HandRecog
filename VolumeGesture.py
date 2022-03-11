from cv2 import cv2
import mediapipe as mp
import time
import numpy as np
import HandModule as hm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

print(volume.GetVolumeRange())


def insertFrame(img1, img2, cx, cy):
    overlay_img1 = np.ones(img1.shape, np.uint8) * 255

    rows, cols, channels = img2.shape
    overlay_img1[cy:rows + cy, cx:cols + cx] = img2

    img2gray = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 220, 55, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    temp1 = cv2.bitwise_and(img1, img1, mask=mask_inv)
    temp2 = cv2.bitwise_and(overlay_img1, overlay_img1, mask=mask)

    result = cv2.add(temp1, temp2)
    return result


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
det=hm.handDetector(detect=0.7)
i=0
while True:
    success, img = cap.read()
    img = det.findHands(img,False)
    lmList=det.findPosition(img)
    if len(lmList)!=0:
        i=i+1
        if i==30:
            i=0

        x1,y1=lmList[4][1], lmList[4][2]
        print(x1,y1)
        x2, y2 = lmList[12][1], lmList[12][2]
        x3, y3 = lmList[0][1], lmList[0][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        #cv2.circle(img, (x1,y1),10,(255,0,255),cv2.FILLED)
        #cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        # cv2.line(img, (x1,y1), (x2,y2),(255,0,255),thickness=5)

        c=np.sqrt((x2-x1)**2+(y2-y1)**2)
        a = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        b = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        alfa=math.acos((a**2+b**2-c**2)/(2*a*b))
        alfa=alfa/np.pi *180
        slika=cv2.imread("strange2.png")

        slika=rotate_image(slika, i*12)
        slika=cv2.resize(slika,(int(c),int(c)))
        posx=x3-int(int(c)/3)
        posy=y3-int(c)
        if posy+int(c)<480 and posx+int(c)<640 and c>40 and posx>0 and posy>0:
            img=insertFrame(img,slika,posx,posy)

        # cv2.putText(img, tekst, (cx, cy), cv2.FONT_HERSHEY_PLAIN,vel,
        # (255, 255, 255), 2)

        # vol=np.interp(alfa,[1,60],[-65.25, 0.0])
        # volume.SetMasterVolumeLevel(vol,None)

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