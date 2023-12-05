import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# Create a dataset.
train_dataset = keras.utils.image_dataset_from_directory(
    './seg_train', batch_size=64, image_size=(150, 150))

for data, labels in train_dataset:
    print('data.shape', data.shape)  # (64, 200, 200, 3)
    print('data.dtype', data.dtype)  # float32
    print('labels.shape', labels.shape)  # (64,)
    print('labels.dtype', labels.dtype)  # int32


model = Sequential([
    Dense(units=16, input_shape=(150, 150, 3), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()
