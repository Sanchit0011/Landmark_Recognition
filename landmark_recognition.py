import os
import pandas as pd
import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
from glob import glob
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

## Create Train, Validation and Test Tensors for CNN
final_data = pd.read_csv('final_data.csv')
final_data = final_data.sort_values(by='landmark_id', ascending=True)
final_data = final_data.reset_index(drop=True)

img_shape = (192, 256) # Image shape (height, width)

def path_to_tensor(img_path):
    
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=img_shape)
    
    # convert PIL.Image.Image type to 3D tensor with shape (192, 256, 3)
    x = image.img_to_array(img)
    
    # convert 3D tensor to 4D tensor with shape (1, 192, 256, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


## Define function to load train, test, and validation datasets
train_path = './train_images/'
valid_path = './valid_images/'
test_path = './test_images/'

def load_dataset(path, train_sample):
    file_out = sorted(glob(path + '*'))
    file_out = np.array([s.replace("\\", "/") for s in file_out])
    
    label_out = pd.Series(name="landmark_id")
    
    for file in file_out:
        filebase = os.path.basename(file)
        name = os.path.splitext(filebase)[0]
        temp = train_sample.landmark_id[train_sample["id"] == name]
        label_out = label_out.append(temp)
        
    label_out = np.array(pd.get_dummies(label_out))
    
    return file_out, label_out

train_file, train_target = load_dataset(train_path, final_data)
valid_file, valid_target = load_dataset(valid_path, final_data)
test_file, test_target = load_dataset(test_path, final_data)

train_tensors = paths_to_tensor(train_file).astype('float32')/255
valid_tensors = paths_to_tensor(valid_file).astype('float32')/255
test_tensors = paths_to_tensor(test_file).astype('float32')/255

## Base CNN model
input_shape = img_shape + (3,)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=4, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='softmax'))

model.summary()

## CNN model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## CNN model training
epochs = 10

checkpointer = ModelCheckpoint(filepath='./saved_models/weights.best.from_baseCNN.hdf5', 
                               verbose=1, save_best_only=True)

hist = model.fit(train_tensors, train_target, 
          validation_data=(valid_tensors, valid_target),
          epochs=epochs, batch_size=64, callbacks=[checkpointer], verbose=1)

## Image data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_
    range=0.2, 
    height_shift_range=0.2, 
    zoom_range=0.3)

valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_tensors, train_target, batch_size= 128)
valid_generator = valid_datagen.flow(valid_tensors, valid_target, batch_size= 128)

## CNN model training after data augmentation
epochs_aug = 50

checkpointer = ModelCheckpoint(filepath='./saved_models/weights.best.from_baseCNN.hdf5', 
                               verbose=1, save_best_only=True)

hist_aug = model.fit_generator(train_generator, steps_per_epoch=6011//128, epochs=epochs_aug,
                    validation_data=valid_generator, validation_steps=751//128,
                    callbacks=[checkpointer], verbose=1)

## Test accuracy
model.load_weights('./saved_models/weights.best.from_baseCNN.hdf5')

landmark_pred = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
landmark_prob = [np.amax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
test_accuracy = np.sum(np.array(landmark_pred) == np.argmax(test_target, axis=1)) / len(landmark_pred)
print('Test accuracy:', test_accuracy)