import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d

from curve_queue import QueueCurve

class Frame:
    def __init__(self):
        self.frameNum = 0
        self.isFirstFrame = True
        self.leftLane = Lane()
        self.rightLane = Lane()
        self.histogram = {}
        self.distanceFromLaneCenter = None
        self.outImage = None
        self.vizMetadata = None
        self.laneWidth = None
        self.verbose_log = False
        self.left_curve_error = False
        self.right_curve_error = False
        self.stored_frame = None
        #self.curveHistCount = curveHistCount
        #self.leftCurveHist = QueueCurve(curveHistCount)
        #self.rightCurveHist = QueueCurve(curveHistCount)

class Lane:
    def __init__(self):
        self.prev = None
        self.fitCurve = None
        self.radius = None
        self.X = None
        self.Y = None



"""
group_lane_pixels()
is used to find and group lane pixels into left and right lane groups
This method does not rely on previous frame

nwindows = the number of sliding windows
margin = the width of the windows +/- margin
minpix = minimum number of pixels found to recenter window
"""
def group_lane_pixels_using_sliding_window(binary_warped, frame,
                                    nwindows = 9, margin = 80, minpix = 50):
    H, W = binary_warped.shape

    # Take a histogram of the bottom half of the image
    # by summing all pixel in a column
    histogram = np.sum(binary_warped[H//2:,:], axis=0)
    histogram = gaussian_filter1d(histogram, 20)
    frame.histogram["curve"] = histogram

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    frame.histogram["leftx_base"] = leftx_base
    frame.histogram["rightx_base"] = rightx_base

    # height of windows - based on nwindows above and image shape
    window_height = np.int(H//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indexes
    left_lane_idxs = []
    right_lane_idxs = []

    frame.vizMetadata = {
        "coords": [],
        "idxs": None
    }

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = H - (window + 1) * window_height
        win_y_high = H - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        frame.vizMetadata["coords"] += [[(win_xleft_low,win_y_low),
                                        (win_xleft_high,win_y_high),
                                        (win_xright_low,win_y_low),
                                        (win_xright_high,win_y_high)]]

        # Identify the nonzero pixels in x and y within the window #
        good_left_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                                    & (nonzerox >= win_xleft_low)
                                    & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                                    & (nonzerox >= win_xright_low)
                                    & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        # If more than 'minpix pixels' were found,
        # recenter next window on their mean position
        new_leftx_current = leftx_current
        new_rightx_current = rightx_current
        if len(good_left_idxs) >= minpix:
            new_leftx_current = np.int(np.mean(nonzerox[good_left_idxs]))
        if len(good_right_idxs) >= minpix:
            new_rightx_current = np.int(np.mean(nonzerox[good_right_idxs]))

        leftx_current = int((leftx_current + new_leftx_current) / 2)
        rightx_current = int((rightx_current + new_rightx_current) / 2)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_idxs = np.concatenate(left_lane_idxs)
        right_lane_idxs = np.concatenate(right_lane_idxs)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("#*50 Error #*50")
        # pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]
    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]
    frame.vizMetadata["idxs"] = (leftx, lefty, rightx, righty)

    return leftx, lefty, rightx, righty


def visualize_sliding_windows(binary_warped, frame):
    # Create an output image to draw on and visualize the result
    H, W = binary_warped.shape
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    radius = 30
    cv2.circle(out_img, (frame.histogram["leftx_base"], H - radius),\
            radius, (255, 255, 0), -1)
    cv2.circle(out_img, (frame.histogram["rightx_base"], H - radius),\
            radius, (0, 255, 255), -1)

    for meta in frame.vizMetadata["coords"]:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, meta[0], meta[1], (0,255,0), 2)
        cv2.rectangle(out_img, meta[2], meta[3], (0,255,0), 2)

    leftx, lefty, rightx, righty = frame.vizMetadata["idxs"]

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    """
    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='blue')
    plt.show()
    plt.close()
    """
    return out_img


def group_lane_pixels_using_prev_frame(binary_warped, frame, margin = 80):
    # margin : the width of the around the previous polynomial to search

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = frame.leftLane.fitCurve
    right_fit = frame.rightLane.fitCurve
    if (frame.verbose_log):
        print("Left  Curve: ", left_fit)
        print("Right Curve: ", right_fit)

    # Set the area of search based on activated x-values ###
    # within the +/- margin of our polynomial function ###
    a, b, c = left_fit[0], left_fit[1], left_fit[2]
    leftX_min = a * (nonzeroy ** 2) + b * nonzeroy + (c - margin)
    leftX_max = a * (nonzeroy ** 2) + b * nonzeroy + (c + margin)
    left_lane_idxs = ((nonzerox > leftX_min) & (nonzerox < leftX_max))

    a, b, c = right_fit[0], right_fit[1], right_fit[2]
    rightX_min = a * (nonzeroy ** 2) + b * nonzeroy + (c - margin)
    rightX_max = a * (nonzeroy ** 2) + b * nonzeroy + (c + margin)
    right_lane_idxs = ((nonzerox > rightX_min) & (nonzerox < rightX_max))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]
    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    # left_lane_idxs, right_lane_idxs would be required for visualization
    frame.vizMetadata =\
                (left_lane_idxs, right_lane_idxs, margin, nonzerox, nonzeroy)

    return leftx, lefty, rightx, righty,


def visualize_search_region_around_prev_lane(binary_warped, frame):

    left_lane_idxs, right_lane_idxs, margin, nonzerox, nonzeroy =\
                                                            frame.vizMetadata
    left_fitx = frame.leftLane.X
    ploty = frame.leftLane.Y
    right_fitx = frame.rightLane.X

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_idxs], nonzerox[left_lane_idxs]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_idxs], nonzerox[right_lane_idxs]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([
                            np.transpose(
                                np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([
                            np.flipud(
                                np.transpose(
                                    np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([
                            np.transpose(
                                np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([
                            np.flipud(
                                np.transpose(
                                    np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    ## End visualization steps ##
    return result


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_curve = np.polyfit(lefty, leftx, 2)
    right_curve = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    # np.linspace(start, stop, num) and stop = num + start - 1
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    # Calc both polynomials using ploty, left_fit and right_fit
    try:
        a, b, c = left_curve[0], left_curve[1], left_curve[2]
        left_fitx = a * ploty**2 + b * ploty + c
    except TypeError:
        print("Error fitting left curve -- using default curve")
        left_fitx = 1 * ploty ** 2 + 1 * ploty

    try:
        a, b, c = right_curve[0], right_curve[1], right_curve[2]
        right_fitx = a * ploty**2 + b * ploty + c
    except TypeError:
        print("Error fitting right curve -- using default curve")
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_curve, right_curve, left_fitx, right_fitx, ploty


def visualize_frame(isFirstFrame, binary_warped, frame):
    if (isFirstFrame):
        return visualize_sliding_windows(binary_warped, frame)
    else:
        return visualize_search_region_around_prev_lane(binary_warped, frame)


def process_frame(binary_warped, frame):
    if (frame.isFirstFrame):
        # Find our lane pixels first
        leftx, lefty, rightx, righty =\
         group_lane_pixels_using_sliding_window(binary_warped, frame)
    else:
        leftx, lefty, rightx, righty =\
        group_lane_pixels_using_prev_frame(binary_warped, frame)

        if leftx.shape[0] == 0 or rightx.shape[0] == 0:
            leftx, lefty, rightx, righty =\
             group_lane_pixels_using_sliding_window(binary_warped, frame)

    #########################################################################
    # Fit a new curve using points obtained by either group lane functions #
    #########################################################################
    left_curve, right_curve, left_fitx, right_fitx, ploty =\
                    fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    """
    frame.leftCurveHist.enqueue(left_curve)
    frame.rightCurveHist.enqueue(right_curve)

    if (frame.frameNum > frame.curveHistCount):
        frame.leftLane.fitCurve = frame.leftCurveHist.mean()
        frame.rightLane.fitCurve = frame.rightCurveHist.mean()
    else:
        frame.leftLane.fitCurve = left_curve
        frame.rightLane.fitCurve = right_curve
    """

    frame.left_curve_error = False
    frame.right_curve_error = False

    if (not frame.isFirstFrame):
        new_left_curve_np = np.array(left_curve[0])
        new_right_curve_np = np.array(right_curve[0])

        old_left_curve_np = np.array(frame.leftLane.fitCurve[0])
        old_right_curve_np = np.array(frame.leftLane.fitCurve[0])

        inc_left_curve = (new_left_curve_np - old_left_curve_np) / old_left_curve_np
        inc_right_curve = (new_right_curve_np - old_right_curve_np) / old_right_curve_np

        tolerance = 50.0
        if (np.abs(inc_left_curve) > tolerance):
            frame.left_curve_error = True

        if (np.abs(inc_right_curve) > tolerance):
            frame.right_curve_error = True

        if (frame.verbose_log):
            print(frame.frameNum, np.abs(inc_left_curve), np.abs(inc_right_curve), frame.left_curve_error, frame.right_curve_error)

    # if (not frame.curve_fit_error):
    if (not frame.left_curve_error):
        frame.leftLane.fitCurve = left_curve
        frame.leftLane.X = left_fitx
        frame.leftLane.Y = ploty

    if (not frame.right_curve_error):
        frame.rightLane.fitCurve = right_curve
        frame.rightLane.X = right_fitx
        frame.rightLane.Y = ploty

    measure_curvature_real_and_car_distance_from_center(frame, binary_warped.shape)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    frame.isFirstFrame = False
    return out_img


# Calculates the curvature of polynomial functions in pixels.
# https://www.intmath.com/applications-differentiation/8-radius-curvature.php
def measure_curvature_real_and_car_distance_from_center(frame, dim):
    if (frame.frameNum != 0 and frame.frameNum % 3 != 0):
        return

    leftx = frame.leftLane.X
    rightx = frame.rightLane.X
    topN = 4
    frame.laneWidth = np.mean(np.abs(leftx[:topN] - rightx[:topN]))
    # Defining conversions in x and y from pixels space to meters
    # Assumptions:
    # Lane length = 30
    # lane min width = 3.7m (U.S. regulation)
    ym_per_pix = 30 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / frame.laneWidth # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
    # here maximum y-value was chosen, corresponding to the bottom of the image
    ploty = frame.leftLane.Y
    y_eval = np.max(ploty)

    left_fit_cr = frame.leftLane.fitCurve
    right_fit_cr = frame.rightLane.fitCurve

    # Calculation of R_curve (radius of curvature)
    a, b, c = left_fit_cr[0], left_fit_cr[1], left_fit_cr[2]
    frame.leftLane.radius = ((1 + (2 * a * y_eval * ym_per_pix + b) ** 2)\
                                ** 1.5) / np.absolute(2 * a)
    a, b, c = right_fit_cr[0], right_fit_cr[1], right_fit_cr[2]
    frame.rightLane.radius = ((1 + (2 * a * y_eval * ym_per_pix + b) ** 2)\
                                ** 1.5) / np.absolute(2 * a)

    # -----------------------------------------------------------------------
    H, W = dim
    image_center = W / 2.0
    lane_center = np.mean((leftx[:topN] + rightx[:topN])) / 2.0
    frame.distanceFromLaneCenter = (image_center - lane_center) * xm_per_pix


"""
binary_warped: thresholded and perspective transformed image (binary birds eye)
"""
def draw_lane(binary_warped, frame, thickness = 10):
    # creating a blank images that will serve as canvas
    zero = np.zeros(binary_warped.shape).astype(np.uint8)
    canvas = np.dstack([zero, zero, zero])

    Y, leftX, rightX = frame.leftLane.Y, frame.leftLane.X, frame.rightLane.X

    cv2_left = np.array([np.transpose(np.vstack([leftX, Y]))])
    cv2_right = np.array([np.flipud(np.transpose(np.vstack([rightX, Y])))])
    cv2_pts = np.hstack((cv2_left, cv2_right))

    cv2.fillPoly(canvas, np.int_([cv2_pts]), (0, 255, 0))
    cv2.polylines(canvas, np.int32([cv2_left]), isClosed = False,
     color = (255, 0, 0), thickness = thickness)
    cv2.polylines(canvas, np.int32([cv2_right]), isClosed = False,
     color = (0, 0, 255), thickness = thickness)
    return canvas


def overlay_lane_region(img, canvas):
    return cv2.addWeighted(img, 1, canvas, 0.5, 0)


def write_lane_data(img, frame):

    """
    cv2.putText(img,
    str(frame.frameNum),
    (1100, 120),
    fontFace = 16,
    fontScale = 2,
    color=(255, 0, 0),
    thickness = 2)
    """

    radius = np.round((frame.leftLane.radius + frame.rightLane.radius) / 2)

    cv2.putText(img,
    'Radius of Curvature {}(m)'.format(radius),
    (60,60),
    fontFace = 16,
    fontScale = 2,
    color=(255,255,255),
    thickness = 2)

    if frame.distanceFromLaneCenter < 0.0:
        str = 'Car is {:03.2f}(m) left of Lane center'.format(-1 * frame.distanceFromLaneCenter)
    else:
        str = 'Car is {:03.2f}(m) right of Lane center'.format(frame.distanceFromLaneCenter)

    cv2.putText(img,
    str,
    (60,120),
    fontFace = 16,
    fontScale = 2,
    color=(255,255,255),
    thickness = 2)

    cv2.putText(img,
    'Lane Width is {:03.2f}(m)'.format(frame.laneWidth),
    (60,180),
    fontFace = 16,
    fontScale = 2,
    color=(255,255,255),
    thickness = 2)

    if (frame.left_curve_error):
        cv2.putText(img,
        'Left Curve Error',
        (60,240),
        fontFace = 16,
        fontScale = 2,
        color=(255, 0, 0),
        thickness = 2)

    if (frame.right_curve_error):
        cv2.putText(img,
        'Right Curve Error',
        (700,240),
        fontFace = 16,
        fontScale = 2,
        color=(255, 0, 0),
        thickness = 2)

    return img
