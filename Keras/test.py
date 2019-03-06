# import tensorflow as tf

# import plaidml.keras
# plaidml.keras.install_backend()

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint

import time
import math

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

from Keras.data_generator import data_generator

def create_model (input_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    # model.add(Conv2D(64, (3, 3), padding='valid', activation='elu'))
    # model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

model = create_model((48, 48, 1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

filepath = "./models/intermediate-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# assuming working directory at project root
train_gen = data_generator('./data/train', './data/train_labels/train_labels.csv', image_size=48, sample_prob=0.1, batch_size=32)
validation_gen = data_generator('./data/train', './data/train_labels/train_labels.csv', image_size=48, sample_prob=0.01, batch_size=1)

model.fit_generator(train_gen,
                    epochs=50,
                    steps_per_epoch=600,
                    validation_data=validation_gen,
                    validation_steps=1000,
                    callbacks=callbacks_list
                    )

model.save_weights('test_model_{}.h5'.format(math.floor(time.time())))