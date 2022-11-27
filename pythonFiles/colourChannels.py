import cv2
import matplotlib.pyplot as plt
img = cv2.imread('dataset\single_prediction\cat_or_dog_1.jpg')
plt.imshow(cv2.imread('dataset\single_prediction\cat_or_dog_1.jpg'))
plt.show()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
plt.show()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))
plt.show()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2LAB))
plt.show()
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
