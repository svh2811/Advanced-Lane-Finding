import numpy as np
import glob
import cv2

from utils import plot_images_along_row, bgr_to_rgb

"""
nx : the number of inside corners along x-axis
ny : the number of inside corners along y-axis
dir: directory contain all chessboard images
     which are going to be used for camera_calibration
"""
def get_camera_calibration_matrix(nx, ny, chessboard_input_dir, visualize = True):

    # Arrays to store object points and image point sfrom all the images
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D point in image plane

    # prepare object points, like
    # (0, 0, 0), (1, 0, 0), (2, 0, 0)...(ny-1, nx-1, 0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)

    imageSize = None
    for img_fname in glob.glob(chessboard_input_dir + "/*.jpg"):
        img = cv2.imread(img_fname)
        imageSize = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if visualize:
                visualize_list = []
                visualize_list.append(("", bgr_to_rgb(np.copy(img))))
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                visualize_list.append(("", bgr_to_rgb(img)))
                plot_images_along_row(visualize_list, figsize = [16, 4])

    return cv2.calibrateCamera(objpoints, imgpoints,
                                (imageSize[1], imageSize[0]), None, None)


def undistort_image(img, cameraMatrix, distCoeffs):
    dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
    return dst


def undistort_images(cameraMatrix, distCoeffs, images_dir, visualize = True):
    for img_fname in glob.glob(images_dir + "/*.jpg"):
        img = cv2.imread(img_fname)
        dst = undistort_image(img, cameraMatrix, distCoeffs)
        if visualize:
            visualize_list = []
            visualize_list.append(("", bgr_to_rgb(img)))
            visualize_list.append(("", bgr_to_rgb(dst)))
            plot_images_along_row(visualize_list, figsize = [16, 4])


def get_perpective_matrices(src, dst):
    # M: perspective matrix
    # src and dst must be floating arrays
    M = cv2.getPerspectiveTransform(src, dst)
    # inverse perspective transform matrix Minv
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


"""
warp image 'img' into new co-ordinate space using matrix T
"""
def warp_image(img, size, T):
    return cv2.warpPerspective(img, T, size, flags=cv2.INTER_LINEAR)
