import cv2
import numpy as np
import imutils

def get_plate_bila(plate, sigma_color, sigma_space):
    screen = None
    try:
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(plate_gray, sigma_color, sigma_space, sigma_space)

        edged = cv2.Canny(filtered, 30, 200)
        edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
        contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    

        cnts = imutils.grab_contours(contours)
        cnts = sorted(cnts, key= cv2.contourArea, reverse= True)[:10]

        

        for c in cnts:
            epsilon = 0.018*cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                screen = approx
                break

        mask = np.zeros(plate_gray.shape, np.uint8)
    
    except:
        pass
    
    return screen

def get_plate_gaus(plate):
    screen = None
    try:
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        filtered = cv2.GaussianBlur(plate_gray, (5, 5), 0)

        edged = cv2.Canny(filtered, 30, 200)
        edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
        contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    

        cnts = imutils.grab_contours(contours)
        cnts = sorted(cnts, key= cv2.contourArea, reverse= True)[:10]

        for c in cnts:
            epsilon = 0.018*cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                screen = approx
                break

        mask = np.zeros(plate_gray.shape, np.uint8)
    
    except:
        pass
    
    return screen