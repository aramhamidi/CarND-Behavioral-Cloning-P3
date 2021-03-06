import cv2
import csv
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

def ResizeImage(image):
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA)
    return image

def augment_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image = np.array(image, dtype = np.float64)
    random_bright = .25+np.random.uniform()         # 0.5
    image[:,:,2] = image[:,:,2]*random_bright
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    return image

def flip_data(image, steering_angle):
    steering_angle = (steering_angle * -0.1)
    image = cv2.flip(image, 1)
    return image, steering_angle


def augment_image(image):
    flip_prob = np.random.random()
    if flip_prob > 0.4:
        image = augment_brightness(image)
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

samples = []
with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del(samples[0])
steering = []
balanced_data = []

for lines in samples:
    angle = float(lines[3])
    if abs(angle) >= 0.05:  #0.25
        steering.append(float(lines[3]))
        balanced_data.append(lines)
    elif np.random.random() > 0.85: #0.9
        steering.append(float(lines[3]))
        balanced_data.append(lines)


samples = balanced_data
print("samples length", len(samples))
print("balanced_data length", len(balanced_data))

plt.hist(steering, bins=[-0.55, -0.5, -0.45, -0.4, -0.35, -0.3,-0.25, -0.2, -0.15, -0.1,-0.05, 0, 0.05, 0.1,0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], align='left')
plt.show()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
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
                
                correction = 0.4
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                if left_angle > 1:
                    lef_angle = 1
                right_angle = center_angle - correction
                if right_angle < -1:
                    right_angle = -1

                center_image = cv2.imread(name1)
                left_image = cv2.imread(name2)
                right_image = cv2.imread(name3)


                #Resize images to 64*64
                center_image = ResizeImage(center_image)
                left_image = ResizeImage(left_image)
                right_image = ResizeImage(right_image)


                #append original images for all three cameras
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                
                #append corrected angles for all three cameras
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

                #flip images and angles
                center_image, center_angle = flip_data(center_image, center_angle)
                left_image, left_angle = flip_data(left_image, left_angle)
                right_image, right_angle = flip_data(right_image, right_angle)

                #append flipped angles
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                
                #append flipped images
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)

                #randomly change the brightness of images
                #Blur all images and change their color-sapce YUV
                for image in images:
                    image = augment_image(image)
    
            X_train = np.array(images)
            y_train = np.array(angles)
            
            (X_train, y_train)= sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print("Data is loaded")  
#------------------------------------Generated Data histogram------------------------------------

num_samples = len(samples)
samples_generated = 0
steering_angles = None
while samples_generated < 6*num_samples:
    X_batch, y_batch = next(train_generator)
    if steering_angles is not None:
        steering_angles = np.concatenate([steering_angles, y_batch])
    else:
        steering_angles = y_batch
    samples_generated += y_batch.shape[0]
 
plt.hist(steering_angles,bins=[-0.55,-0.5, -0.45, -0.4, -0.35, -0.3,-0.25, -0.2, -0.15, -0.1,-0.05,0,0.05, 0.1,0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], align='left')
plt.show()

#------------------------------------Model Architecture------------------------------------------
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Input
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers.pooling import AveragePooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3), output_shape=(64,64,3)))
model.add(Cropping2D(cropping=((13,0), (0,0)), input_shape=(64,64,3)))
model.add(Convolution2D(24,3,3, subsample=(2,2), activation="elu"))#5
#model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(36,3,3, subsample=(2,2), activation="elu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,3,3, subsample=(2,2), activation="elu"))
#model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(64,1,1, activation="elu"))#3
#model.add(MaxPooling2D())
model.add(Convolution2D(64,1,1, activation="elu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()


##------------------------------Comiling Model----------------------------------##

model.compile(loss='mse', optimizer='adam')

# Once the model is compiled we'll train the feature and label arrays
# we just build.
# WE also shuffle the data and also split off 20% of the data to use 
# for validation set.

history_object = model.fit_generator(train_generator, samples_per_epoch =
    6*len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=2, verbose=1)

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
