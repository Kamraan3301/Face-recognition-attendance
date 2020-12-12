

                                                                ### PART 1 (DEMO)
# import cv2
# import face_recognition

# imgtrain = face_recognition.load_image_file('/home/harun/kamran_paid/1_jJih-QYQXa6Gj7eaN3svbA.jpeg')
# imgtrain = cv2.cvtColor(imgtrain,cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('/home/harun/kamran_paid/1318517-kg.jpg')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
# face_location = face_recognition.face_locations(imgtrain)[0] #aita muloto aikhane face distance measure kore, such us tomar amar chockher and face er landmarks er distance.
# encodekorlam = face_recognition.face_encodings(imgtrain)[0] 
# cv2.rectangle(imgtrain,(face_location[3],face_location[0]),(face_location[1],face_location[2]),(255,0,255),2)
 
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
# results = face_recognition.compare_faces([encodekorlam],encodeTest)
# faceDis = face_recognition.face_distance([encodekorlam],encodeTest)
# print(results,faceDis)
# cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
# cv2.imshow('muhtadi_Original',imgtrain)
# cv2.imshow('Muhtadi_test',imgTest)
# cv2.waitKey(0)



#part2

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# from PIL import ImageGrab
 
path = '/home/kamran/paidprojectopencv/ml-project/Images'
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

def markAttendance(name):
     with open('/home/kamran/paidprojectopencv/ml-project/Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



n=findEncodings(images)
print(len(n))
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        comparing_face = face_recognition.compare_faces(n,encodeFace)
        facesimilrities = face_recognition.face_distance(n,encodeFace)
        # print(facesimilrities)
        matchIndex= np.argmin(facesimilrities)
        ooaa=min(facesimilrities)*100

        
        ooaa=100-(ooaa)+20
        ooaa=round(ooaa,2)
        oopp=[]
        oopp.append(ooaa)
        # print(oopp)
        ox=''.join(map(str,oopp))

        if comparing_face [matchIndex]:
            name= classNames[matchIndex].upper()
            
            print(name)  
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1-6,y2+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.putText(img,ox,(x2-6,y2+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)



##csv er entry with cam 0
#csv er exit with cam 1
##csv concatenate
##make functions for FLASK
##route reg and log deploy
## deploy csv btn conn with sql
##csv cal
##csv edit access through admin
##all route
##frontend visualization
##Extras will be added later

