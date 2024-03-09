import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import cvzone

model = YOLO('best100_v4.pt')

area1 = [(413, 333), (689, 263), (738, 282), (464, 384)]
area2 = [(467, 392), (788, 289), (831, 309), (469, 432)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('/Users/thanaboon/Desktop/Rmutt/year3.2/AI/project_detect/video.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker1 = Tracker()
tracker2 = Tracker()


cy1=294
cy2=276

counter1 =  []
counter2= []
offset= 4

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1080, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []
    female = []
    list2 = []
    male = []
    

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]


    
        if 'Female' in c :
            list1.append([x1,y1,x2,y2])
            female.append(c)
        elif 'Male' in c:
            list2.append([x1,y1,x2,y2])
            male.append(c)
            
    bbox1_idx = tracker1.update(list1)
    bbox2_idx = tracker2.update(list2)
    
    for bbox1 in bbox1_idx:
        for f in female:
            x3, y3, x4, y4, id1 = bbox1
            # ใช้จุดขวาล่างของ bounding box เป็นตำแหน่งจุด
            cxm= x4
            cym= y4
            if cym<(cy1+offset) and cym>(cy1-offset):
                cv2.circle(frame, (cxm, cym), 4, (0, 255, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)
                if counter1.count(id1) == 0:
                    counter1.append(id1)
#################### Male ###################

    for bbox2 in bbox2_idx:
        for m in male:
            x5, y5, x6, y6, id2 = bbox2
            # ใช้จุดขวาล่างของ bounding box เป็นตำแหน่งจุด
            cxc= x6 
            cyc= y6
            if cyc <(cy1+offset) and cyc > (cy1-offset):
                cv2.circle(frame, (cxc, cyc), 4, (0, 255, 0), -1)
                cv2.rectangle(frame, (x5, y5), (x6, y6), (0, 0, 255), 1)
                cvzone.putTextRect(frame, f'{id2}', (x5, y5), 1, 1)
                if counter2.count(id2) == 0:
                    counter2.append(id2)
              


            
    cv2.line(frame,(379,cy1),(824,cy1),(0,0,255),2)

    femalelen = (len(counter1))
    malelen = (len(counter2))
    cvzone.putTextRect(frame, f'Female Count: {femalelen}', (19,30), 2,1)
    cvzone.putTextRect(frame, f'Male Count: {malelen}', (18,71), 2,1)


    #cv2.putText(frame, str('2'), (482, 440), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)



    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
