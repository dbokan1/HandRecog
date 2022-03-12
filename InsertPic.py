from cv2 import cv2
import numpy as np



def insertFrame(img1, img2, cx, cy):
    overlay_img1 = np.ones(img1.shape, np.uint8) * 255

    rows, cols, channels = img2.shape
    overlay_img1[cx:rows + cx, cy:cols + cy] = img2

    img2gray = cv2.cvtColor(overlay_img1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 230, 60, cv2.THRESH_BINARY_INV)
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


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    slika = cv2.imread('src/strange2.png')
    slika=rotate_image(slika,45)
    slika = cv2.resize(slika, (200, 200))
    rez=insertFrame(img,slika,10,10)
    cv2.imshow("Image", rez)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()