#######################################
# writen by FETTAH Taha & SEBBAH hala
#######################################

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from keras import applications
from keras.layers import GlobalAveragePooling2D,Dropout , Dense 
from keras.models import Model


# In[2]:


CATEGORIES = []
imgSize = 64
Data =[]

DATADIR = "C:/Users/hala/Downloads/carsData/Dacia/daciaNet"
listDir  = os.listdir(DATADIR)
for category in listDir:
    CATEGORIES.append(category)
    


# In[3]:


def create_training_data():
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                image= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                if((image is None)==False):
                    newImage = cv2.resize(image , (imgSize , imgSize))
                    Data.append([newImage , class_num])
            except Exception as e:
                pass


# In[4]:


Data = []
trainingData = []
testData = []
create_training_data()
numClasse = 0
index1 = 0
index2=0
while(index2<len(Data)):
    while((index2<len(Data)and(Data[index2][1]==numClasse))):
        index2+=1
    indexTemp = index1+int((index2-index1)*0.8)
    for j in range (index1,indexTemp):
        trainingData.append(Data[j])
    for j in range (indexTemp,index2):
        testData.append(Data[j])
    index1 = index2
    numClasse+=1 


# In[5]:


random.shuffle(trainingData)


# In[6]:


featuresTrain = []
labelsTrain = []
featuresTest =[]
labelsTest = []


# In[7]:


for feature , label in trainingData:
    featuresTrain.append(feature)
    labelsTrain.append(label)
for feature , label in testData:
    featuresTest.append(feature)
    labelsTest.append(label)


# In[8]:

featuresTrain = np.array(featuresTrain).reshape(-1,imgSize,imgSize,1)
labelsTrain = np.array(labelsTrain)

featuresTest = np.array(featuresTest).reshape(-1,imgSize,imgSize,1)
labelsTest = np.array(labelsTest)


# In[9]:


ari= []
for i in range(len(labelsTrain)):
    if(labelsTrain[i]==0):
        ari.append([1,0,0,0,0,0])
    elif (labelsTrain[i]==1):
        ari.append([0,1,0,0,0,0])
    elif (labelsTrain[i]==2):
        ari.append([0,0,1,0,0,0])
    elif (labelsTrain[i]==3):
        ari.append([0,0,0,1,0,0])
    elif (labelsTrain[i]==4):
        ari.append([0,0,0,0,1,0])
    elif (labelsTrain[i]==5):
        ari.append([0,0,0,0,0,1])
labelsTrain = np.array(ari)
#Y_train = convert_to_one_hot(labelsTrain, 6).T
#Y_test = convert_to_one_hot(labelsTest, 6).T


# In[10]:


featuresTrain = tf.keras.utils.normalize(featuresTrain, axis=1)  # scales data between 0 and 1
featuresTest = tf.keras.utils.normalize(featuresTest, axis=1) 


# In[11]:


img_height,img_width = 64,64 
num_classes = 6
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,1))


# In[12]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)


# In[13]:


from keras.optimizers import SGD, Adam
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(featuresTrain, labelsTrain, epochs = 100, batch_size = 64)


######
# evaluate and save model


preds = model.evaluate(featuresTest,labelsTest)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.save('daciaModel.h5') 
