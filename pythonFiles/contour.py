import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('dataset\single_prediction\cat_or_dog_1.jpg')
canny = cv2.Canny(img,100,200)


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
contours,hierarchies = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
blank = np.zeros(img.shape,dtype='uint8')
plt.imshow(cv2.drawContours(blank,contours,-1,(0,0,255),2))
plt.show()