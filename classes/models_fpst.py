import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

import numpy as np



def get_basic_cnn(kernels, input_shape, output_shape):
    model = keras.Sequential()

    model.add(Input(shape=(input_shape[0], input_shape[1], 1)))

    # Convolutional Layers
    for kernel in kernels:
        model.add(Conv2D(kernel, (3, 3), activation='relu', padding='same'))

    # Upsampling Layers
    upscale_factor = int(output_shape[0] / input_shape[0])
    upscale_times = int(np.log2(upscale_factor))
    for i in range(upscale_times):
        model.add(UpSampling2D((2, 2)))

    # Output Layer
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    return model