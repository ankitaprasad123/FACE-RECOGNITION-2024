import cv2
import numpy as np
import sqlite3

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');             #to detect the face in camera
cam=cv2.VideoCapture(0)        #0 is for web camera


def insertorupdate(Id,Name,Age):                            #function is for sqlite database
    conn=sqlite3.connect('sqlite.db')                              #database connect
    cmd="SELECT * FROM PERSON WHERE ID="+str(Id)                   #table name is Person
    cursor=conn.execute(cmd)                                         #cursor to execute statement
    isRecordExist=0                    # assume there is no record in our table
    for row in cursor:
        isRecordExist=1
    if (isRecordExist == 1):                      #if there is a record exist in our table
        conn.execute("UPDATE PERSON SET Name=? WHERE Id=?", (Name, Id))
        conn.execute("UPDATE PERSON SET Age=? WHERE Id=?", (Age, Id))
    else:
        conn.execute("INSERT INTO PERSON (Id,Name,Age) VALUES(?,?,?)", (Id, Name, Age))

    conn.commit()
    conn.close()
#insert user defined  value into table

Id=input('Enter Id of person you want to insert')
Name=input('Enter Name of person you want to insert')
Age=input('Enter Age of person you want to insert')

insertorupdate(Id,Name,Age)


#detect face in web camera coding
sampleNum=0;                                                       #assume there is no samples in dataset
while (True):
   ret,img =cam.read();                                                 #open camera
   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);                           #image convert to bgrgray color
   faces=faceDetect.detectMultiScale(gray,1.3,5);  #scal face
   for (x,y,w,h) in faces:
        sampleNum=sampleNum+1;    #if face is detected increment
        cv2.imwrite("dataset/user."+str(Id)+" . "+str(sampleNum)+".jpg",gray[y:y+h,x:x+w]); #to create a dataset file when we  our open camera and when we stop our camera our id and our picture automatically will be save in our dataset folder
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100); #delay time
   cv2.imshow('Face',img)
   cv2.waitKey(1);
   if(sampleNum>20):
        break;
cam.release()
cv2.destroyAllWindows()
#quit


