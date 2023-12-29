from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import detect_plate

plate_detector = YOLO("model/best.pt")

# load photo
img = cv2.imread("saved_image.jpg")
# img = cv2.resize(img, (640, 640))

detections = plate_detector.predict(img)

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

for detection in detections_:
    (x1, y1, x2, y2) = detection
    roi = img[int(y1):int(y2), int(x1):int(x2)]

    plate = detect_plate.get_plate(roi)

    # if plate is not None:
    #     x3, y3 = plate[0][0][0], plate[0][0][1]             #top left
    #     x4, y4 = plate[1][0][0], plate[1][0][1]             #top right
    #     x5, y5 = plate[2][0][0], plate[2][0][1]             #bottom right
    #     x6, y6 = plate[3][0][0], plate[3][0][1]             #bottom left

    #     cv2.drawContours(roi, [plate], -1, (0, 255, 0), 3)

    #     cv2.circle(roi, (x3, y3), 5, (0, 0, 255), -1)
    #     cv2.circle(roi, (x4, y4), 5, (0, 0, 255), -1)
    #     cv2.circle(roi, (x5, y5), 5, (0, 0, 255), -1)
    #     cv2.circle(roi, (x6, y6), 5, (0, 0, 255), -1)
    cv2.drawContours(roi, [plate], -1, (0, 255, 0), 3)
        

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()