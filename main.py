from ultralytics import YOLO
import cv2
import numpy as np
import os
from utils.tracker import Tracker
import pandas as pd
from utils.get_exact import *
from utils.warp_perspective import *

print(os.listdir())

# load model
car_detector = YOLO("model/yolov8s.pt")
plate_detector = YOLO("model/vn-large-8m/best.pt")

vehicle_tracker = Tracker()

cap = cv2.VideoCapture("./videos/sample.mp4")

# using YOLO we have to define the classes we want to detect
# here, firstly, we have to detect cars, trucks or buses
# vehicle = [2, 5, 7]    

count = 0

frame_nmr = -1
ret = True
while ret:
    if not ret:
        break
    frame_nmr += 1
    if frame_nmr % 50 == 0:
        continue
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

        cv2.putText(frame, str(track_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
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

            x5 = x5 - 2
            y5 = y5 - 2
            x6 = x6 + 2
            y6 = y6 + 2
            # cv2.putText(roi, str(class_id), (x3, y5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            roi_plate = roi_car[int(y5):int(y6), int(x5):int(x6)]

            plate = extract_plate(roi_plate)

            cv2.rectangle(roi_car, (x5, y5), (x6, y6), (255, 0, 0), 3)

            if plate is not None and check_contour(plate):
                warped = warpPerspective(roi_plate, plate)
                cv2.imshow("warped", warped)
                
                # apply OCR here
            
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()