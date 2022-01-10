import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

df = pd.DataFrame(
    columns=["Roll_No","Date","Time","Attendence"])
path = 'ImageAttendence'
roll_no=[]
date_1=[]
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

def markAttendance(name):
   
    now=datetime.now()
    date=now.strftime("%d-%m-%y")
    time=now.strftime("%H:%M:%S")
    if (name not in roll_no) | (date not in date_1) :
        roll_no.append(name)
        date_1.append(date)
        df.loc[len(df)] = [name, date, time, "p"]
    df.to_csv("attendance_data/Attendance.csv")
    


cap = cv2.VideoCapture(0)
# address="http://192.168.1.12:8080/video"
# cap.open(address)

while (cap.isOpened()):
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
