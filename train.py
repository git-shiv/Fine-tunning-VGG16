import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Flatten, Lambda,Input, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline

Image_Size = [224,224]
train_path = 'C:/Users/ASUS/Projects/Cone/train'
valid_path='C:/Users/ASUS/Projects/Cone/valid'

vgg = VGG16(input_shape= Image_Size + [3], weights= 'imagenet', include_top= False)

for layers in vgg.layers:
    layers.trainable= False
    
floders = glob ('C:/Users/ASUS/Projects/Cone/train/*')

x = Flatten()(vgg.output)
prediction = Dense(1,activation = 'softmax')(x)

    
model= Model(inputs= vgg.input, outputs= prediction)


model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001), metrics = ['accuracy'])

train_datagen= ImageDataGenerator(rescale= 1./255,
                                 shear_range= 0.2,
                                 zoom_range=0.2,
                                 horizontal_flip= True)

test_datagen= ImageDataGenerator(rescale= 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/ASUS/Projects/Cone/train', target_size=(224,224),
                                                  batch_size= 32,
                                                  class_mode='categorical')

test_set = test_datagen.flow_from_directory('C:/Users/ASUS/Projects/Cone/valid', target_size=(224,224),
                                                  batch_size= 32,
                                                  class_mode='categorical')


r= model.fit(validation_data = test_set,
            epochs=5,
              steps_per_epoch=len(training_set),
              validation_steps=len(test_set),
              x=training_set,
          verbose=2
                      )

plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['acc'],label = 'train acc')
plt.plot(r.history['val_acc'],label = 'val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_loss')

model.save('C:/Users/ASUS/Projects/models/Conefeatured_new_model.h5')
