from cv2 import cv2
import mediapipe as mp
import time


########################################
# Basic use of mediapipe hand and face recognition
########################################

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils
faceDetect=mp.solutions.face_detection
face=faceDetect.FaceDetection(0.75)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rezHand = hands.process(imgRGB)

    if rezHand.multi_hand_landmarks:
        for handLms in rezHand.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    rezFace=face.process(img)
    if rezFace.detections:
        for id, x in enumerate(rezFace.detections):
            #mpDraw.draw_detection(img,x)
            box=x.location_data.relative_bounding_box
            h,w,c=img.shape
            box2= int(box.xmin*w), int(box.ymin*h), int(box.width*w), int(box.height*h)
            cv2.rectangle(img,box2, (255,0,255),2)

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