import cv2

img = cv2.imread("./data/test_images/straight_lines1.jpg")
cv2.imshow("Get Co-Ords", img)
cv2.waitKey()
