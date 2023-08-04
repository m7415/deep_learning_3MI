import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout

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

def get_sub_pixel(input_shape, output_shape, dropout_rate=0.0):

    input = Input(shape=(input_shape[0], input_shape[1], 1))

    # feature identification
    x = Conv2D(64, (5, 5), activation='tanh', padding='same', kernel_initializer='orthogonal')(input)
    dropout = keras.layers.Dropout(dropout_rate)(x)

    # finer feature identification
    x = Conv2D(32, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(dropout)
    x = Conv2D(32, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(x)
    dropout = keras.layers.Dropout(dropout_rate)(x)

    # sub-pixel convolution
    upscale_factor = int(output_shape[0] / input_shape[0])
    x = Conv2D(upscale_factor ** 2, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(dropout)
    dropout = keras.layers.Dropout(dropout_rate)(x)
    
    output = tf.nn.depth_to_space(dropout, upscale_factor)

    model = Model(inputs=input, outputs=output)    

    return model

def get_deeper_sub_pixel(input_shape, output_shape, dropout_rate=0.0):
    
    input = Input(shape=(input_shape[0], input_shape[1], 1))

    # feature identification
    x = Conv2D(128, (5, 5), activation='tanh', padding='same', kernel_initializer='orthogonal')(input)
    x = Conv2D(128, (5, 5), activation='tanh', padding='same', kernel_initializer='orthogonal')(x)
    dropout = keras.layers.Dropout(dropout_rate)(x)

    # finer feature identification
    x = Conv2D(64, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(dropout)
    x = Conv2D(64, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(x)
    dropout = keras.layers.Dropout(dropout_rate)(x)

    x = Conv2D(32, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(x)
    x = Conv2D(32, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(x)
    dropout = keras.layers.Dropout(dropout_rate)(x)

    # sub-pixel convolution
    upscale_factor = int(output_shape[0] / input_shape[0])
    x = Conv2D(upscale_factor ** 2, (3, 3), activation='tanh', padding='same', kernel_initializer='orthogonal')(dropout)
    dropout = keras.layers.Dropout(dropout_rate)(x)

    output = tf.nn.depth_to_space(dropout, upscale_factor)

    model = Model(inputs=input, outputs=output)

    return model

# An implementation of the SRCNN model (SRCNN: Super-Resolution Convolutional Neural Network)
def get_SRCNN(data_shape):
    weight_decay = 0.00005
    regu = keras.regularizers.l2(weight_decay)
    dropout_rate = 0.2
    init = 'he_normal'
    # the input has the same shape as the output because it has been upscaled with bicubic interpolation
    input = Input(shape=(data_shape[0], data_shape[1], 1))

    # improve the quality of the image with convolutional layers
    x = Conv2D(64, (9, 9), padding='same', kernel_initializer=init, kernel_regularizer=regu)(input)
    #x = BatchNormalization()(x)  # Add batch normalization after the first convolutional layer
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)  # Add dropout after the first convolutional layer

    x = Conv2D(32, (1, 1), padding='same', kernel_initializer=init, kernel_regularizer=regu)(x)
    #x = BatchNormalization()(x)  # Add batch normalization after the second convolutional layer
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)  # Add dropout after the second convolutional layer

    output = Conv2D(1, (5, 5), activation='sigmoid', padding='same', kernel_initializer=init, kernel_regularizer=regu)(x)

    model = Model(inputs=input, outputs=output)

    return model
