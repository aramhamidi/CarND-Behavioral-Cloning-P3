import cv2
import csv
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV) 
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


samples = []
with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del(samples[0])
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=64):
    num_samples = len(samples)
    
    while 1:
        
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                
                name1 = 'data/IMG/' + batch_sample[0].split('/')[-1]
                name2 = 'data/IMG/' + batch_sample[1].split('/')[-1]
                name3 = 'data/IMG/' + batch_sample[2].split('/')[-1]
                
                center_image = cv2.imread(name1)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image = augment_brightness(center_image)
                center_image = cv2.GaussianBlur(center_image, (3,3), 0)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2YUV)
                #images.append(center_image)
                
                center_angle = float(batch_sample[3])
                
                
                left_image = cv2.imread(name2)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_image = augment_brightness(left_image)
                left_image = cv2.GaussianBlur(left_image, (3,3), 0)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2YUV)
                #images.append(left_image)
                
                left_angle = center_angle + 0.25
                
                
                right_image = cv2.imread(name3)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_image = augment_brightness(right_image)
                right_image = cv2.GaussianBlur(right_image, (3,3), 0)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2YUV)
                #images.append(right_image)
                
                right_angle = center_angle - 0.25
                
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                
                #augmented_images, augmented_angles = [], []
                #for image, angle in zip(images, angles):
                #   images.append(image)
                #  angles.append(angle)
                #  images.append(cv2.flip(image,1))
                #  angles.append(angle * -1.0)
                
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle * -1.0)
                
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle * -1.0)
                
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle * -1.0)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            (X_train, y_train)= sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Data is loaded")  

# num_samples = len(samples)
# samples_generated = 0
# steering_angles = None
# while samples_generated < num_samples:
#     X_batch, y_batch = next(train_generator)
#     if steering_angles is not None:
#         steering_angles = np.concatenate([steering_angles, y_batch])
#     else:
#         steering_angles = y_batch
#     samples_generated += y_batch.shape[0]
# 
# plt.hist(steering_angles)
# plt.show()


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers.pooling import AveragePooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3),output_shape=(160,320,3)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="elu"))
model.add(Dropout(0.3))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="elu"))
#model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="elu"))
model.add(Dropout(0.3))
model.add(Convolution2D(64,3,3, activation="elu"))
#model.add(Dropout(0.4))
model.add(Convolution2D(64,3,3, activation="elu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# model.summary()
# For the loss function we use mean square , or MSE, which is differnt
# than cross_entropy function, again becuase this is a regression network 
# not a classifer network.

# We want to minimize the error between the stearing measurementss that the 
# Network predicts and the ground truth steering measurement.
# MSE is good loss function for this.

model.compile(loss='mse', optimizer='adam')

# Once the model is compiled we'll train the feature and label arrays
# we just build.
# WE also shuffle the data and also split off 20% of the data to use 
# for validation set.

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
# Now we will save the model and download it onto our local machine,
# and see if it works for driving the simulator.

model.save('model.h5')
print("model saved!")
exit()
