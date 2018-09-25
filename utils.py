import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile

from moviepy.editor import VideoFileClip


def extract_video_frames(video_file_path, video_length, ts,
                            video_frame_save_dir):
    video = VideoFileClip(video_file_path)  # frame_rate = 25

    for t_s in np.arange(0.0, video_length, ts):
        # save frame at t = t_s as JPEG
        video.save_frame('video_frame_save_dir/{:02d}.jpg'\
                            .format(int(t_s)), t=t_s)


def color_in_range(image, low_threshold, high_threshold):
    mask = cv2.inRange(image, low_threshold, high_threshold)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


def get_plot_as_np_array(plt, display_axis = True):
    fileName = tempfile.gettempdir() + "/plot.png"
    plt.rcParams["figure.figsize"] = [50, 30]
    plt.tight_layout()
    if not display_axis:
        plt.axis('off')
    plt.savefig(fileName)
    plt.close()
    return bgr_to_rgb(cv2.imread(fileName))


"""
images: list of tuples, where each tuple has image title
        and the image itself
"""
def plot_images_along_row(images, figsize = [19, 8]):
    plt.rcParams["figure.figsize"] = figsize
    plt.tight_layout()
    fig = plt.figure()
    n = len(images)
    for i in range(n):
        a = fig.add_subplot(1, n, i+1)

        if len(images[i]) == 3:
            if images[i][2]:
                a.axis("off")
        if (len(images[i][1].shape) == 2):
            imgplot = plt.imshow(images[i][1], cmap='gray',\
                                            vmin = 0, vmax = 1)
        else:
            imgplot = plt.imshow(images[i][1])
        a.set_title(images[i][0])
    plt.show()
    plt.close()


def bgr_to_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def gauss_blur(img, kernel):
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def median_blur(img, kernel):
    return cv2.medianBlur(img, kernel)


def get_file_name_from_path(path):
    return path[path.rfind("/") + 1 : path.rfind(".")]
