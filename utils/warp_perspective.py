import cv2
import numpy as np

def arrange(corners):
    # store the top-left, top-right, bottom-right, bottom-left of the plate respectively
    corners = np.array(corners)
    corners = np.reshape(corners, (4, 2))
    
    right = corners[corners[:,0].argsort()[::-1]][:2]
    left = corners[corners[:,0].argsort()][:2]

    top_left = left[left[:,1].argsort()][0]
    bottom_left = left[left[:,1].argsort()[::-1]][0]
    top_right = right[right[:,1].argsort()][0]
    bottom_right = right[right[:,1].argsort()[::-1]][0]

    corners = np.array([top_left, top_right, bottom_right, bottom_left])
    return corners

def warpPerspective(image, corners):
    # rearrange the corners in top_left, top_right, bottom_right, bottom_left order
    corners = arrange(corners)

    # calculate the width and height of the plate
    h = corners[3][1] - corners[0][1]
    w = corners[1][0] - corners[0][0]

    # top_left, top_right, bottom_right, bottom_left
    try:
        top_left = corners[0]
    except:
        top_left = (0, 0)

    try:    
        top_right = corners[1]
    except:
        top_right = (image.shape[1], 0)

    try:
        bottom_right = corners[2]
    except:
        bottom_right = (image.shape[1], image.shape[0])
    try:
        bottom_left = corners[3]
    except:
        bottom_left = (0, image.shape[0])

    if w > (3 * h):
        dst_size = (520, 110)
    elif (3 * w) > h > (2 * w):
        dst_size = (330, 165)
    elif (2 * w) > h:
        dst_size = (190, 140)

    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(image, M, dst_size)
    return warped