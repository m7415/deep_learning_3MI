import tensorflow as tf
from tensorflow import keras
from keras import layers

# define a Unet class

class Unet:
    def __init__(self, input_shape = (3, 512, 512), output_shape = (1, 512, 512)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def summary(self, graph = False):
        self.model.summary()
        if graph:
            keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True)

    def save_model(self, model_path):
        self.model.save(model_path)
    
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
    
    def train(self, train_data, train_label, epochs = 10, batch_size = 1):
        self.model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def build_model(self):
        # Contracting Path
        inputs = layers.Input(shape=self.input_shape)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

        # Expansive Path
        up4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3)
        up4 = layers.concatenate([up4, conv2])
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

        up5 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
        up5 = layers.concatenate([up5, conv1])
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model