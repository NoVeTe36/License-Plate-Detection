import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import imutils

def detect_corner(plate):
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(plate_gray, 5, 250, 250)

    edged = cv2.Canny(filtered, 30, 200) 
    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    

    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key= cv2.contourArea, reverse= True)[:10]

    screen = None

    for c in cnts:
        epsilon = 0.018*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            screen = approx
            break

    mask = np.zeros(plate_gray.shape, np.uint8)

    new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)
    new_img = cv2.bitwise_and(plate, plate, mask= mask)

    (x,y) = np.where(mask == 255)

    (topx, topy) = (np.min(x),np.min(y))
    (bottomx, bottomy) = (np.max(x),np.max(y))

    croped = plate_gray[topx:bottomx + 1, topy:bottomy + 1]

    cv2.imshow("Plate", croped)
    cv2.imshow("Plate", new_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = cv2.imread('./images/roi_1.jpg')
    # resize image to fit the model yolo v8
    
    model = YOLO("model/vn-kaggle/best.pt")
    detections = model.predict(image)
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

    for detection in detections_:
        (x1, y1, x2, y2) = detection

        cv2.rectangle(image, (x1, y1), (x2, y2), (111, 111, 111), 3)

        roi = image[int(y1):int(y2), int(x1):int(x2)]

        plate = detect_corner(roi)

