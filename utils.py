import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



"""
nx : the number of inside corners along x-axis
ny : the number of inside corners along y-axis
dir: directory contain all chessboard images
     which are going to be used for camera_calibration
"""
def get_camera_calibration_matrix(nx, ny,
    chessboard_input_dir, chessboard_output_dir = None):

    # Arrays to store object points and image point sfrom all the images
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D point in image plane

    # prepare object points, like
    # (0, 0, 0), (1, 0, 0), (2, 0, 0)...(ny-1, nx-1, 0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)

    images = glob.glob(chessboard_input_dir + "/*.jpg")
    imageSize = None

    for img_fname in images:
        fname = get_file_name_from_path(img_fname)
        print("Calibrating Image: ", fname)
        img = cv2.imread(img_fname)
        imageSize = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.drawChessboardCorners
            if chessboard_output_dir is not None:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imwrite(chessboard_output_dir + "/" + fname + ".jpg", img)

    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.calibrateCamera
    return cv2.calibrateCamera(objpoints, imgpoints,
                                (imageSize[1], imageSize[0]), None, None)


def undistort_images(cameraMatrix, distCoeffs,
                    images_dir, output_images_dir = None):
    images = glob.glob(images_dir + "/*.jpg")
    for img_fname in images:
        fname = get_file_name_from_path(img_fname)
        print("Undistorting Image: ", fname)
        img = cv2.imread(img_fname)
        # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.undistort
        dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
        if output_images_dir is not None:
            cv2.imwrite(output_images_dir + "/" + fname + ".jpg", dst)


def gradient_thresh(img, orient="x", sobel_kernel=3, thresh=(20, 100)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    abs_sobel = np.absolute(abs_sobel)
    """
    It's not entirely necessary to convert to 8-bit (range from 0 to 255)
    but in practice, it can be useful in the event that we've written a
    function to apply a particular threshold, and if we want it to work the
    same on input images of different scales, like jpg vs. png.
    we could just as well choose a different standard range of values,
    like 0 to 1 etc.
    """
    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(binary_output)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

































def plot_images(img1_loc, img2_loc, img1_title, img2_title, fileName):
    plt.rcParams["figure.figsize"] = [19, 8]
    plt.tight_layout()
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    img = mpimg.imread(img1_loc)
    imgplot = plt.imshow(img)
    a.set_title(img1_title)
    a = fig.add_subplot(1,2,2)
    img = mpimg.imread(img2_loc)
    imgplot = plt.imshow(img)
    a.set_title(img2_title)
    plt.show()
    fig.savefig(fileName, bbox_inches = 'tight')
    plt.close()


def get_file_name_from_path(path):
    return path[path.rfind("/") + 1 : path.rfind(".")]
