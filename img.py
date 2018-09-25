import matplotlib.pyplot as plt
import numpy as np
import cv2
from thresholding import *

# list of lists
def plot_images_along_row(images):
    fig = plt.figure()

    rows = len(images)
    cols = len(images[0])

    i = 0
    for row in range(rows):
        for col in range(cols):
            a = fig.add_subplot(rows, cols, i+1)
            if (len(images[row][col][1].shape) == 2):
                imgplot = plt.imshow(images[row][col][1], cmap='gray')
            else:
                imgplot = plt.imshow(images[row][col][1])
            a.set_title(images[row][col][0])
            i += 1

    plt.show()
    plt.close()


img = cv2.imread("challenge_video_frames/02.jpg")

#"""
colorspace1 = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

channels1 = [
    ("L", colorspace1[:, :, 0]),
    ("u", colorspace1[:, :, 1]),
    ("v", colorspace1[:, :, 2])
]

#"""
"""
colorspace2 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

channels2 = [
    ("L", colorspace2[:, :, 0]),
    ("a", colorspace2[:, :, 1]),
    ("b", colorspace2[:, :, 2])
]

colorspace3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

channels3 = [
    ("H", colorspace3[:, :, 0]),
    ("S", colorspace3[:, :, 1]),
    ("V", colorspace3[:, :, 2])
]


colorspace4 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

channels4 = [
    ("H", colorspace4[:, :, 0]),
    ("L", colorspace4[:, :, 1]),
    ("S", colorspace4[:, :, 2])
]

"""
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gradx = gradient_thresh(rgb_img, orient="x", sobel_kernel=7, thresh=(8, 16))

grady = gradient_thresh(rgb_img, orient="y", sobel_kernel=3, thresh=(20, 100))

sobel_grads = [
    ("gray", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
    ("gX", gradx),
    ("gY", grady)
]

mag_thresh_img = mag_thresh(rgb_img, sobel_kernel=3, mag_thresh=(20, 200))

mean_gX = cv2.medianBlur(gradx, 5)
dir_thresh_img = dir_threshold(rgb_img, sobel_kernel=3, thresh=(np.pi/2, 2*np.pi/3))

others = [
    ("Og Img", rgb_img),
    ("mag", mag_thresh_img),
    ("mean_gx", mean_gX)
]

plot_images_along_row([others, channels1, sobel_grads])

#plot_images_along_row([channels1, channels2, channels3, channels4])
