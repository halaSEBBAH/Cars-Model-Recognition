# Cars-Model-Recognition
a system to classify and recognise cars models using deep learning and computer vision


### Prerequisites

pyhton - 
OpenCv - 
keras 

### Step 1 : Data collectionning and cleansing

all data we used is in here


### Step 2 : Logo recognition using SIFT algorithm 

```
# predefined SIFT function in OpenCV library 
sift = cv2.xfeatures2d.SIFT_create()

```




### Step 3 : image recognition using deep learning and computer vision
we used convolutionnal neural networks to make the classifiers , as deep neural networks are actually very popular and aknowledged to be performant in classification tasks . 

<img src="Images/CNN.PNG">


As the tendacy goes for making layers deeper and deeper , we used  **ResNet50** initialy with empt weights and trained it on the data we gathered 

##### CNN model using ResNet50 architecture
our models construction and training looks like this in the three Models we built

```
img_height,img_width = 64,64 
num_classes = 6

base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,1))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import SGD, Adam
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(featuresTrain, labelsTrain, epochs = 100, batch_size = 64)

```
#### why ResNet50


## Acknowledgments
Our professor : M. RACHID OULAD HAJ THAMI

## Authors
* **SEBBAH Hala** 
* **FETTAH Taha** 
