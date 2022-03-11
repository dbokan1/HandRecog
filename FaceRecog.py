from cv2 import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
faceDetect=mp.solutions.face_mesh
face=faceDetect.FaceMesh(max_num_faces=2)
draw=mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rez=face.process(imgRGB)

    if rez.multi_face_landmarks:
        for  x in rez.multi_face_landmarks:
            draw.draw_landmarks(img,x,faceDetect.FACE_CONNECTIONS,
                                draw.DrawingSpec(thickness=1, circle_radius=1))

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
