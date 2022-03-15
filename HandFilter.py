from cv2 import cv2
import time
import numpy as np
import HandModule as hm

#########################################
# Responsive instagram video filter,
# changes size and orientation to accomodate hand position
#########################################

##########################################
# Volume gesture control modules
##########################################
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))



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


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    det=hm.handDetector(detect=0.7)
    i=0
    shearing=0
    thumbx1, thumby1, middlex1, middley1, palmx1, palmy1, pinkyx1, pinkyy1=0,0,0,0,0,0,0,0

    while True:
        success, img = cap.read()
        img = det.findHands(img,False)

        for j in range(det.handNum(img)):
            lmList=det.findPosition(img,j)

            if len(lmList)!=0:
                i=i+1
                if i==30:
                    i=0

                thumbx, thumby= lmList[4][1], lmList[4][2]
                middlex, middley = lmList[12][1], lmList[12][2]
                palmx, palmy = lmList[0][1], lmList[0][2]
                pinkyx, pinkyy = lmList[20][1], lmList[20][2]
                c=np.sqrt((middlex - thumbx) ** 2 + (middley - thumby) ** 2)
                a = np.sqrt((middlex - palmx) ** 2 + (middley - palmy) ** 2)
                b = np.sqrt((palmx - thumbx) ** 2 + (palmy - thumby) ** 2)

                # cx,cy= (thumbx + middlex) // 2, (thumby + middley) // 2
                # cv2.circle(img, (x1,y1),10,(255,0,255),cv2.FILLED)
                # cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                # cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                # cv2.line(img, (x1,y1), (x2,y2),(255,0,255),thickness=5)
                # alfa=math.acos((a**2+b**2-c**2)/(2*a*b))
                # alfa=alfa/np.pi *180

                slika=cv2.imread("src/strange2.png")

                slika=rotate_image(slika, i*12)
                slika = cv2.resize(slika, (int(a), int(a)))

                num_cols, num_rows, kanali= slika.shape
                src_points = np.float32([[pinkyx1, pinkyy1], [thumbx1, thumby1], [middlex1, middley1]])
                dst_points = np.float32([[pinkyx, pinkyy], [thumbx, thumby], [middlex, middley]])
                matrix = cv2.getAffineTransform(src_points, dst_points)

                # Warping the picture to follow hand movement and orientation
                if shearing==1:
                    slika = cv2.warpAffine(slika, matrix, (num_cols, num_rows))

                # Setting the baseline finger positions for affine shearing
                if shearing==0 and c>300:
                    thumbx1, thumby1, middlex1, middley1, pinkyx1, pinkyy1 = thumbx, thumby, middlex, middley, pinkyx, pinkyy
                    shearing=1


                posx= int((middlex + palmx) / 2) - int(int(a) / 2)
                posy= int((middley + palmy) / 2) - int(int(a) / 2)
                if posy+int(a)<480 and posx+int(a)<640 and c>60 and posx>0 and posy>0:
                    img=insertFrame(img,slika,posx,posy)

                # Volume gesture control commands
                # vol=np.interp(alfa,[1,60],[-65.25, 0.0])
                # volume.SetMasterVolumeLevel(vol,None)

            else:
                shearing=0

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

main()

