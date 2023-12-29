from ultralytics import YOLO
import cv2
import numpy as np
import os
from tracker import Tracker
import pandas as pd
from utils.contour_plate import *

print(os.listdir())

# load model
car_detector = YOLO("model/yolov8s.pt")
plate_detector = YOLO("model/vn-large-8m/best.pt")

vehicle_tracker = Tracker()

# load video 1 frame per 2 seconds
cap = cv2.VideoCapture("./videos/sample.mp4")

# using YOLO we have to define the classes we want to detect
# here, firstly, we have to detect cars, trucks or buses
# vehicle = [2, 5, 7]    

count = 0

# read video 1 frame per second

frame_nmr = -1
ret = True
while ret:
    if not ret:
        break
    frame_nmr += 1
    # if frame_nmr % 50 == 0:
    #     continue
    ret, frame = cap.read()
    detections = car_detector.predict(frame)
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
        if confidence > 0.5:
            detections_.append([x1, y1, x2, y2])
            
    trackings = vehicle_tracker.update(detections_)
    
    for tracking in trackings:
        
        (x3, y3, x4, y4, track_id) = tracking

        # cv2.putText(frame, str(class_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (111, 111, 111), 3)

        roi_car = frame[int(y3):int(y4), int(x3):int(x4)]

        detections_plate = plate_detector.predict(roi_car)
        b = detections_plate[0].boxes.data
        px_ = pd.DataFrame(b.cpu().numpy())
        detections_plate_ = []

        for index, detection in px_.iterrows():
            x5 = int(detection[0])
            y5 = int(detection[1])
            x6 = int(detection[2])
            y6 = int(detection[3])
            confidence = float(detection[4])
            class_id = int(detection[5])
            if confidence > 0.5:
                detections_plate_.append([x5, y5, x6, y6])

        for detection in detections_plate_:
            x5 = int(detection[0])
            y5 = int(detection[1])
            x6 = int(detection[2])
            y6 = int(detection[3])
            # cv2.putText(roi, str(class_id), (x3, y5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            roi_plate = roi_car[int(y5):int(y6), int(x5):int(x6)]
            
            # plate = get_plate_gaus(roi_plate)

            # if plate is not None:
            #     cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
            # else:
            #     plate = get_plate_bila(roi_plate, 13, 15)
            #     if plate is not None:
            #         cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
            #     else:
            #         plate = get_plate_bila(roi_plate, 5, 250)
            #         if plate is not None:
            #             cv2.drawContours(roi_plate, [plate], -1, (0, 255, 0), 3)
            #         else:
            #             cv2.rectangle(roi_plate, (x5, y5), (x6, y6), (0, 0, 255), 3)

            cv2.rectangle(roi_car, (x5, y5), (x6, y6), (0, 0, 255), 3)            
            
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
