import cv2

def check_contour(corners):
    """Check if contour is valid."""
    if corners is None:
        return False
    if len(corners) < 3:
        return False    
    if cv2.contourArea(corners) < 200:
        return False
    return True