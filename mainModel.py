import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import visualkeras
from glob import glob
from tqdm import tqdm
import cv2
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,BatchNormalization,Input,Flatten,Dropout,GlobalMaxPooling2D
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

df=pd.DataFrame()
label=0
# image_data=[]
# labels=[]
for category in glob('./input/Garbage_Collective_Data/*'):
    for files in tqdm(glob(category+'/*')):
        image=cv2.imread(files)
        image=cv2.resize(image,(224,224))    #As images are of different shape so resizing all to (224,224,3)
        data=pd.DataFrame({'image':[image],'label':[label]})
        df=pd.concat([df, data])
    label+=1

celltext=[]
transform_classes={}
label=0
for name in glob('./input/Garbage_Collective_Data/*'):
    names=[]
    names.append(name.split('/')[-1].upper().replace('_',' '))
    names.append(label)
    celltext.append(names)
    transform_classes[label]=name.split('/')[-1]
    print("names: ", names)
    print("label: ", label)
    label+=1

#fig, ax = plt.subplots()

#plt.axis('off')
#plt.axis('tight')
#column_names=['Categories','Label']
#row_names=[x for x in range(1,13)]
#plot=plt.table(cellText=celltext,cellLoc='center',cellColours=[['#b3ff1a','#00ffff']]*12,
#               rowLabels=row_names,rowColours=['#ff2021']*12,colWidths=[1.2]*10,colLabels=column_names,loc='center')
#plot.set_fontsize(24)
#plot.scale(1, 4)
#plt.show()

print("Voy a Sample")
df=df.sample(frac=1).reset_index(drop=True)

print("Voy a head")
df.head()

print("Voy a info")
df.info()

print("Voy a iloc")
df.image.iloc[1].shape

print("Voy a sklearn")
from sklearn.model_selection import train_test_split
train_image,test_image,train_label,test_label=train_test_split(df.image,df.label,test_size=0.01,random_state=42)


#plt.figure(figsize=(20,15))
#for i in range(22):
#    plt.subplot(5,4,(i%20)+1)
#    rand_idx=np.random.randint(50)
#    plt.title('{0} waste'.format(transform_classes[train_label.iloc[rand_idx]]),fontdict={'size':10})
#    plt.axis('off')
#    plt.imshow(train_image.iloc[rand_idx])
#     plt.tight_layout()
#plt.show()

def change_to_input_dimension(data):
    data=np.reshape(data.to_list(),(len(data),224,224,3))
    return data
train_image=change_to_input_dimension(train_image)
test_image=change_to_input_dimension(test_image)

early_stop=EarlyStopping(monitor='val_loss',patience=4)
reduceLR=ReduceLROnPlateau(patience=3)

model=Sequential()
model.add(Input(shape=(224,224,3)))
model.add(Conv2D(256,(3,3)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(GlobalMaxPooling2D())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

r=model.fit(train_image,train_label,validation_data=(test_image,test_label),epochs=30,callbacks=[early_stop,reduceLR])


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('modelOrdenado30.tflite', 'wb') as f:
  f.write(tflite_model)

plt.figure(figsize=(10,8))
plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.title('ACCURACY',fontdict={'size':22})
plt.show()
