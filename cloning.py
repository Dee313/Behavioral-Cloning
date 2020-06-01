import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Folder where training data and images are stored
data_path = './drive_data/'
image_path = data_path + 'IMG/'

angle_adjustment = 0.1

num_left_turns = 0
num_right_turns = 0
num_straights = 0

images = []
angles = []

# Augment the data by flipping the images and corresponding steering angles
# Analyze the steering angles and split into left, right and straight driving
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        center_image = cv2.imread(image_path + line[0].split('/')[-1])
        # Convert to RGB for drive.py
        center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        images.append(center_image_rgb)
        angles.append(float(line[3]))
        # Flip the image
        images.append(cv2.flip(center_image_rgb, 1))
        angles.append(-float(line[3]))

        left_image = cv2.imread(image_path + line[1].split('/')[-1])
        # Convert to RGB for drive.py
        left_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        images.append(left_image_rgb)
        angles.append(float(line[3])+angle_adjustment)
        # Flip the image
        images.append(cv2.flip(left_image_rgb, 1))
        angles.append(-(float(line[3])+angle_adjustment))

        right_image = cv2.imread(image_path + line[2].split('/')[-1])
        # Convert to RGB for drive.py
        right_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        images.append(right_image_rgb)
        angles.append(float(line[3])-angle_adjustment)
        # Flip the image
        images.append(cv2.flip(right_image_rgb, 1))
        angles.append(-(float(line[3])-angle_adjustment))

        if(float(line[3]) < -0.12):
            num_left_turns += 1
        elif(float(line[3]) > 0.12):
            num_right_turns += 1
        else:
            num_straights += 1

print("Raw data")
print("Left turns = ", num_left_turns)
print("Right turns = ", num_right_turns)
print("Straights = ", num_straights)

X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

train_samples_size = len(X_train)
validation_samples_size = len(X_val)

num_left_turns = 0
num_right_turns = 0
num_straights = 0

for train_sample in y_train:
    if(float(train_sample) < -0.12):
        num_left_turns += 1
    elif(float(train_sample) > 0.12):
        num_right_turns += 1
    else:
        num_straights += 1

print("Trained data")   
print("Left turns = ", num_left_turns)
print("Right turns = ", num_right_turns)
print("Straights = ", num_straights)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=9, shuffle=True)

model.save('model.h5')
