import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('./final/model/yolov8n.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cap=cv2.VideoCapture('./final/sample.mp4')

count=0

tracker=Tracker()

cy1=322
cy2=368
offset=6

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1

    detections = model(frame)[0]
        
#    print(px)
    list=[]
             
    for row in detections.boxes.data.tolist():
        
    #    print(row)
    #    break
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[4])
        c=int(row[5])
        if c==3 or c==5 or c==7:
            list.append([x1,y1,x2,y2,d])
    bbox_id=tracker.update(list)
    
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
           


#    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
#    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.imshow("RGB", frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()