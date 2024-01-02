import cv2
import numpy as np
from utils.contour_plate import *
from utils.hough_plate import *
from utils.warp_perspective import *

# def extract_plate(roi_plate):
#     plate = None
#     plate = get_plate_gaus(roi_plate)
#     if plate is not None and check_contour(plate):
#         # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
#         warped = warpPerspective(roi_plate, plate)
#         return warped
#     else:
#         plate = get_plate_bila(roi_plate, 13, 15)
#         if plate is not None and check_contour(plate):
#             warped = warpPerspective(roi_plate, plate)
#             # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
#             return warped
#         else:
#             plate = get_plate_bila(roi_plate, 5, 250)
#             if plate is not None and check_contour(plate):
#                 # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
#                 warped = warpPerspective(roi_plate, plate)
#                 return warped
#             else:
#                 plate = hough_plate(roi_plate)
#                 if plate is not None and check_contour(plate):
#                     # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
#                     warped = warpPerspective(roi_plate, plate)
#                     return warped
#                 else:
#                     # if no contour is found, take the corners of the roi_plate as the corners of the plate
#                     plate = np.array([[0, 0], [roi_plate.shape[1], 0], [roi_plate.shape[1], roi_plate.shape[0]], [0, roi_plate.shape[0]]])
#                     warped = warpPerspective(roi_plate, plate)
#     cv2.rectangle(roi_plate, (0, 0), (roi_plate.shape[1], roi_plate.shape[0]), (0, 0, 255), 3)
#     return plate

def extract_plate(roi_plate):
    plate = None
    plate = get_plate_gaus(roi_plate)
    if plate is not None and check_contour(plate):
        # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
        
        return plate
    else:
        plate = get_plate_bila(roi_plate, 13, 15)
        if plate is not None and check_contour(plate):
            
            # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
            return plate
        else:
            plate = get_plate_bila(roi_plate, 5, 250)
            if plate is not None and check_contour(plate):
                # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
                
                return plate
            else:
                plate = hough_plate(roi_plate)
                if plate is not None and check_contour(plate):
                    # cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
                    
                    return plate
                else:
                    # if no contour is found, take the corners of the roi_plate as the corners of the plate
                    plate = np.array([[0, 0], [roi_plate.shape[1], 0], [roi_plate.shape[1], roi_plate.shape[0]], [0, roi_plate.shape[0]]])
    cv2.rectangle(roi_plate, (0, 0), (roi_plate.shape[1], roi_plate.shape[0]), (0, 0, 255), 3)
    return plate