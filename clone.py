import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os.path

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32, augment=False):
# Note: If augment is True, this generator will actually return 4 times the specified batch size
    num_samples = len(samples)
    correction = 0.15
    datadir = 'data/'

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Center
                center_image = cv2.imread(datadir + batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                if augment:
                    # Center flipped
                    center_flipped = np.fliplr(center_image)
                    flipped_angle = -center_angle
                    images.append(center_flipped)
                    angles.append(flipped_angle)

                    # Side cameras
                    left_image = cv2.imread(datadir + batch_sample[1].lstrip())
                    left_angle = center_angle + correction
                    right_image = cv2.imread(datadir + batch_sample[2].lstrip())
                    right_angle = center_angle - correction

                    images.append(left_image)
                    images.append(right_image)
                    angles.append(left_angle)
                    angles.append(right_angle)

            X = np.array(images)
            y = np.array(angles)

            yield shuffle(X, y)


train_generator = generator(train_lines, batch_size=32, augment=True)
valid_generator = generator(validation_lines, batch_size=32, augment=False)

from keras.models import Sequential
from keras.layers import AveragePooling2D, Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, Activation, Dropout


drop_rate = 0.2;
model = Sequential()

# Crop and normalize
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3), name='crop'))  # output (90, 320, 3)
model.add(AveragePooling2D(name='downsample'))                                          # output (45, 160, 3)
model.add(Lambda(lambda x: (x/255.0) - 0.5, name='normalize'))

# First Conv + Pooling block
model.add(Convolution2D(32, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv1_1')) # output (45, 160, 32)
model.add(Convolution2D(32, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv1_2'))
model.add(MaxPooling2D(border_mode='valid', name='pool1'))                   # output (23, 80, 32)


# Second Conv + Pooling block
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv2_1')) # output (23, 80, 64)
model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv2_2'))
model.add(MaxPooling2D(border_mode='valid', name='pool2'))                  # output (12, 40, 64)


# Third Conv + Pooling block
model.add(Convolution2D(128, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv3_1')) # output (12, 40, 128)
model.add(Convolution2D(128, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv3_2'))
model.add(MaxPooling2D(border_mode='valid', name='pool3'))                    # output (6, 20, 128)


# Fourth Conv + Pooling block
model.add(Convolution2D(256, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv4_1')) # output (6, 20, 256)
model.add(Convolution2D(256, 3, 3, subsample=(1,1), border_mode='same', activation='relu', name='conv4_2'))
model.add(MaxPooling2D(border_mode='valid', name='pool4'))                   # output (3, 10, 256)


# Flatten
model.add(Flatten(name='flat'))

# Fully Connected + relu
model.add(Dense(4096, name='fc'))
model.add(Dropout(drop_rate))
model.add(Activation('relu', name='activation'))
model.add(Dense(1, name='output'))

train_samples = len(train_lines) * 4
validation_samples = len(validation_lines)

output_file = 'model.h5'

if os.path.isfile(output_file):
    model.load_weights(output_file, by_name=True)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=train_samples, validation_data=valid_generator, nb_val_samples=len(validation_lines), nb_epoch=7)

model.save(output_file)
