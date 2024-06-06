import os                       #READ AND WRITE FILE
import cv2                      #OPEN CAMERA
import numpy as np              #ARRAY
from PIL import Image          #IMAGE FILE READ AND WRITE

#cv2 is  Package which  recognize our faces in camera
#recognizer=cv2.Face.LBPHFaceRecognizer_create()
recognizer=cv2.face.LBPHFaceRecognizer_create()
path="dataset"




def get_images_with_id(path):
    images_paths = [os.path.join(path,f) for f in os.listdir(path)]   #set images path to os
    faces=[]
    ids=[]
    for single_image_path in images_paths:
        #L stands for luminamce it is nothing but increasing the brightness in the images.
        faceImg=Image.open(single_image_path).convert('L')  #converting each image into black and white images
        faceNp=np.array(faceImg,np.uint8)
        id=int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(ids),faces
ids,faces=get_images_with_id(path)
recognizer.train(faces,ids)
recognizer.save("recognizer/trainingdata.yml") #tained faces with their id
cv2.destroyAllWindows()

