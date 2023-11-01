import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, LeakyReLU, UpSampling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization, Dense, Flatten, Concatenate, AveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

# Basic, hand-crafted super-resolution models

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

# Pre-upsampling super-resolution models

def get_srcnn(input_shape):
    input = Input(shape=(input_shape[0], input_shape[1], 1))

    # patch extraction and representation
    conv1 = Conv2D(64, (9, 9), activation='relu', padding='same')(input)
    
    # non-linear mapping
    conv2 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv1)

    # reconstruction
    output = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(conv2)

    model = Model(input, output)

    return model

def get_vdsr(input_shape):
    input = Input(shape=(input_shape[0], input_shape[1], 1))
    
    # feature extraction
    for i in range(18):
        if i == 0:
            conv = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
        else:
            conv = Conv2D(64, (3, 3), activation='relu', padding='same')(conv)
    
    # non-linear mapping
    conv = Conv2D(64, (1, 1), activation='relu', padding='same')(conv)

    # reconstruction
    output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv)

    model = Model(input, output)

    return model


# Post-upsampling super-resolution models

def get_fsrcnn(input_shape, upscale_factor, d, s, m):
    input = Input(shape=(input_shape[0], input_shape[1], 1))

    # Feature Extraction
    conv1 = Conv2D(d, (5, 5), activation='relu', padding='same')(input)

    # Shrinking
    conv2 = Conv2D(s, (1, 1), activation='relu', padding='same')(conv1)

    # Non-linear Mapping
    for i in range(m):
        if i == 0:
            conv3 = Conv2D(s, (3, 3), activation='relu', padding='same')(conv2)
        else:
            conv3 = Conv2D(s, (3, 3), activation='relu', padding='same')(conv3)

    # Expanding
    conv4 = Conv2D(d, (1, 1), activation='relu', padding='same')(conv3)

    # Deconvolution (Upsampling to input size * upscale_factor)
    output = Conv2DTranspose(1, (9, 9), strides=(upscale_factor, upscale_factor), activation='sigmoid', padding='same')(conv4)

    model = Model(input, output)

    return model

def train(model, X_train, y_train, batch_size, epochs, verbose=True):
    # Adjusting patience for early stopping to be coherent with around 20 epochs
    early_stopping_patience = max(1, int(epochs * 0.2))

    # define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1, restore_best_weights=True)
    ]

    # train the model with validation split
    if verbose:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks, verbose=1)
    else:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks, verbose=0)

    return history

def train_and_evaluate_model(X_train, y_train, X_test, y_test, batch_size, epochs, num_layers, num_filters_1, num_filters_2, activation, batch_norm, dropout, init):
    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = get_espcn(input_shape, 2, num_layers, num_filters_1, num_filters_2, activation, batch_norm, dropout, init)

    # Compile the model
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    history = train(model, X_train, y_train, batch_size, epochs, verbose=False)

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    return loss, history

def conv_block(x, num_filters, kernel_size=(3, 3), activation='relu', batch_norm=False, dropout=0.0, init='he_normal'):
    x = Conv2D(num_filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer=init)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    if activation == 'leaky_relu':
        x = LeakyReLU(alpha=0.2)(x)
    elif activation == 'relu':
        x = Activation('relu')(x)
    else:
        raise ValueError('Activation function not supported')
    return x

# init to test : he_normal, he_uniform, glorot_normal, glorot_uniform, lecun_normal, lecun_uniform
# dropout rate to test : 0.0, 0.2, 0.4, 0.6, 0.8

# this beauty here !!!
def get_espcn(input_shape, upscale_factor, num_layers, num_filters_1, num_filters_2, activation, batch_norm, dropout, init):
    input = Input(shape=(input_shape[0], input_shape[1], 1))

    # Feature Map Extraction with skip connections
    for i in range(num_layers):
        if i == 0:
            conv = conv_block(input, num_filters_1, kernel_size=(5, 5), activation=activation, batch_norm=batch_norm, dropout=dropout, init=init)
            skip = conv
        else:
            conv = conv_block(conv, num_filters_2, kernel_size=(3, 3), activation=activation, batch_norm=batch_norm, dropout=dropout, init=init)

    # Skip connection
    conv = Concatenate()([conv, skip])

    # Sub-pixel convolution
    conv = Conv2D(upscale_factor ** 2, (3, 3), padding='same', kernel_initializer=init)(conv)
    output = tf.nn.depth_to_space(conv, upscale_factor)

    model = Model(input, output)

    return model

def get_discriminator(input_shape):
    # create a classifier model
    input = Input(shape=(input_shape[0], input_shape[1], 1))

    # Feature Extraction and downsampling 
    # the conv blocks have padding='same' by default so the output size is the same as the input size
    # to reduce the size by half, we then have to use downsampling (strides=(2, 2) doesn't work)
    conv1 = conv_block_leaky(input, 64, kernel_size=(3, 3), strides=(1, 1))
    down1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block_leaky(down1, 128, kernel_size=(3, 3), strides=(1, 1))
    down2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block_leaky(down2, 256, kernel_size=(3, 3), strides=(1, 1))
    down3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block_leaky(down3, 512, kernel_size=(3, 3), strides=(1, 1))
    down4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    # Classification
    flat = Flatten()(down4)
    dense1 = Dense(1024, activation='relu')(flat)
    dense2 = Dense(1, activation='sigmoid')(dense1)

    model = Model(input, dense2)

    return model

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train_gan(generator, discriminator, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate=1e-3):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history = {
        'loss': [],
        'val_loss': [],
    }

    @tf.function
    def train_step(images):
        (X, y) = images

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(X, training=True)

            real_output = discriminator(y, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # train the generator and discriminator
    print('Training GAN...')
    print()
    for epoch in range(epochs):
        start = time.time()

        for i in range(len(X_train) // batch_size):
            X_batch = X_train[i*batch_size:min((i+1)*batch_size, len(X_train))]
            y_batch = y_train[i*batch_size:min((i+1)*batch_size, len(y_train))]
            train_step((X_batch, y_batch))
            # remove the previous line from the terminal
            sys.stdout.write("\r\033[K")
            print(f'Epoch {epoch + 1} - {i+1}/{len(X_train) // batch_size}', end='')

        # add training loss to history
        """ mse_loss = tf.keras.losses.MeanSquaredError()(y_train, generator(X_train))
        history['loss'].append(mse_loss.numpy())
        print(f' - loss: {history["loss"][-1]:.2e}', end='') """
        # add validation loss to history
        mse_loss = tf.keras.losses.MeanSquaredError()(y_test, generator(X_test))
        history['val_loss'].append(mse_loss.numpy())
        print(f' - val_loss: {history["val_loss"][-1]:.2e}')

    return history