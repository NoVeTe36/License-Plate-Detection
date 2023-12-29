from ultralytics import YOLO
import cv2
import numpy as np
import os
from tracker import Tracker
import pandas as pd

print(os.listdir())

# load model
car_detector = YOLO("model/yolov8n.pt")
plate_detector = YOLO("model/vn-large/best.pt")

vehicle_tracker = Tracker()

# load video 1 frame per 2 seconds
cap = cv2.VideoCapture("./sample.mp4")

# using YOLO we have to define the classes we want to detect
# here, firstly, we have to detect cars, trucks or buses

vehicle = [2, 5, 7]    
count = 0

# read video 1 frame per second

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
        if int(class_id) in vehicle:
            detections_.append([x1, y1, x2, y2])
            
    trackings = vehicle_tracker.update(detections_)
    
    for tracking in trackings:
        
        (x3, y3, x4, y4, track_id) = tracking

        cv2.putText(frame, str(class_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (111, 111, 111), 3)

        roi = frame[int(y3):int(y4), int(x3):int(x4)]

        detections_plate = plate_detector.predict(roi)
        b = detections_plate[0].boxes.data
        px_ = pd.DataFrame(b.cpu().numpy())
        detections_plate_ = []

        for index, detection in px_.iterrows():
            x3 = int(detection[0])
            y3 = int(detection[1])
            x4 = int(detection[2])
            y4 = int(detection[3])
            confidence = float(detection[4])
            class_id = int(detection[5])
            if int(class_id) == 0:
                detections_plate_.append([x3, y3, x4, y4])

        for detection in detections_plate_:
            x3 = int(detection[0])
            y3 = int(detection[1])
            x4 = int(detection[2])
            y4 = int(detection[3])
            cv2.rectangle(roi, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(roi, str(class_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            
    cv2.imshow("frame", frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
