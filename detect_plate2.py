import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

def preprocess_image(image):
    # Convert to Grayscale Image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.fastNlMeansDenoising(gray_image, None, 20, 7, 21)

    # Canny Edge Detection
    canny_edge = cv2.Canny(gray_image, 170, 200)

    # Morphological operations
    kernel = np.ones((5, 15), np.uint8)
    canny_edge = cv2.dilate(canny_edge, kernel)
    canny_edge = cv2.erode(canny_edge, kernel)

    return canny_edge

def get_plate(image):
    # Find contours based on Edges
    contours, new = cv2.findContours(
        image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    # Initialize license Plate contour and x,y coordinates
    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None

    # Find the contour with 4 potential corners and creat ROI around it
    for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4:  # see whether it is a Rect
            print("Got ctr")
            contour_with_license_plate = approx            
    
    if contour_with_license_plate is not None:
        return contour_with_license_plate
    else:
        return None
    
def main():
    # Load original image
    original_image = cv2.imread('./images/test.jpg')

    # Preprocess image
    canny_edge = preprocess_image(original_image)

    # YOLO object detection
    model = YOLO("model/vn-large/best.pt")
    detections = model.predict(original_image)
    a = detections[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy())
    detections_ = []
    for index, detection in px.iterrows():
        x1 = int(detection[0])
        y1 = int(detection[1])
        x2 = int(detection[2])
        y2 = int(detection[3])
        confidence = float(detection[4])
        class_id = int(detection[5])
        detections_.append([x1, y1, x2, y2])

    if len(detections_) == 0:
        print("No plate detected")

    # Process YOLO detections
    for detection in detections_:
        (x1, y1, x2, y2) = detection

        # Adjust coordinates to the original image
        x1 += int(detection[0])
        y1 += int(detection[1])
        x2 += int(detection[0])
        y2 += int(detection[1])

        cv2.rectangle(original_image, (x1, y1), (x2, y2), (111, 111, 111), 3)

        roi = original_image[y1:y2, x1:x2]

        # Get the license plate coordinates in the original image
        plate_contour = get_plate(canny_edge[y1:y2, x1:x2])
        if plate_contour is not None:
            # Adjust the contour coordinates to the original image
            plate_contour[:, 0, 0] += x1
            plate_contour[:, 0, 1] += y1
            cv2.drawContours(original_image, [plate_contour], -1, (0, 255, 0), 3)
        else: 
            print("No edge detected")

    cv2.imshow("Detected Plates", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
