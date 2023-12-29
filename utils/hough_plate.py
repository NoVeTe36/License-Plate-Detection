import cv2
import numpy as np
from utils.hough import HoughBundler
import math


def slope(x1,y1,x2,y2):
    ###finding slope
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

def drawLine(image,x1,y1,x2,y2):

    m=slope(x1,y1,x2,y2)
    h,w=image.shape[:2]
    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (255, 255, 255), 1)


def rearrange_points(corners):
    # Find center
    center = [0]*2
    for i in range(corners.shape[0]):
        center[0] += corners[i][0][0]
        center[1] += corners[i][0][1]
        center[0] /= 4
        center[1] /= 4

    rearranged = [None]*4
    for i in range(4):
        if corners[i][0][0] < center[0] and corners[i][0][1] > center[1]:
            rearranged[i] = 0
        elif corners[i][0][0] > center[0] and corners[i][0][1] > center[1]:
            rearranged[i] = 1
        elif corners[i][0][0] > center[0] and corners[i][0][1] < center[1]:
            rearranged[i] = 2
        elif corners[i][0][0] < center[0] and corners[i][0][1] < center[1]:
            rearranged[i] = 3

    corners_copy = [None]*4
    for i in range(4):
        corners_copy[rearranged[i]] = [corners[i][0].tolist()]
    return corners_copy

def hough_plate(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    edge = cv2.Canny(img_blur, 30, 200)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

    # houghlines
    lines = cv2.HoughLinesP(edge, 1, np.pi/180, 60, minLineLength=10, maxLineGap=10)

    lines = HoughBundler(min_distance=10,min_angle=5).process_lines(lines)

    length = np.array([[None]*2 for i in range(lines.shape[0])])
    for i in range(len(length)):
        length[i][0] = math.sqrt( (lines[i][0][2]-lines[i][0][0])**2 + (lines[i][0][3] - lines[i][0][1])**2 )
        length[i][1] = i
    length_copy = np.flip(length[length[:,0].argsort()])[:4,0]
    lines_copy = np.array([None]*4)
    for i in range(lines_copy.shape[0]):
        lines_copy[i] = lines[length_copy[i]]
    lines = lines_copy

    edge_copy = np.zeros_like(edge)

    for i in range(4):
        drawLine(edge_copy, lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])

    corners = cv2.convexHull(np.array(cv2.goodFeaturesToTrack(edge_copy,4,0.01,10), dtype = np.int32), False)

    #   corners = np.float32(rearrange_points(corners))

    # rearrange the points
    corners = np.uint8(rearrange_points(corners))

    bottom_left_x = corners[0][0][0]
    bottom_left_y = corners[0][0][1]

    bottom_right_x = corners[1][0][0]
    bottom_right_y = corners[1][0][1]

    top_right_x = corners[2][0][0]
    top_right_y = corners[2][0][1]

    top_left_x = corners[3][0][0]
    top_left_y = corners[3][0][1]

    corners = np.array([[[bottom_left_x, bottom_left_y]], [[bottom_right_x, bottom_right_y]], [[top_right_x, top_right_y]], [[top_left_x, top_left_y]]], dtype = np.int32)
    return corners