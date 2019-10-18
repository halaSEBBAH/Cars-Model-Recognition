#######################################
# writen by FETTAH Taha & SEBBAH hala
#######################################


#write descriptors in  files that we name fileDes and fileKp for dacia
#you have to create a folder specified in the  variable DATADIR
import picklerit
import os
import numpy as np
import cv2
import os



DATADIR = "C:/Users/Dell/Desktop/Logo dacia"
#os.chdir('C:/Users/Dell/Desktop/Liste de descripteurs dacia')
sift = cv2.xfeatures2d.SIFT_create()
listDir = os.listdir(DATADIR) 


#write descriptors in fileDes
liste = []
liste2 = []
cpt = 0
with open('C:/Users/Dell/Desktop/Liste de descripteurs dacia/fileDes', 'wb') as file:
    pickler = pickle.Pickler(file)
    for imageName in listDir:
        img = cv2.imread(os.path.join(DATADIR,imageName),cv2.IMREAD_GRAYSCALE)
        kp , des = sift.detectAndCompute(img , None)
        pickler.dump(des)
        liste.append(des)
        cpt+=1


#write keypoints in fileKp after serializing it because picker can't dump cv2.KeyPoint
liste = []
with open('C:/Users/Dell/Desktop/Liste de descripteurs dacia/fileKp', 'wb') as file:
    pickler = pickle.Pickler(file)
    for imageName in listDir:
        img = cv2.imread(os.path.join(DATADIR,imageName),cv2.IMREAD_GRAYSCALE)
        kp , des = sift.detectAndCompute(img , None)
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            liste.append(temp)
        pickler.dump(liste)
        
        liste = [] 
