import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adadelta
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(
        self,
        input_shape,
        isConv=False,
        hidden_units=None,
        filters=None,
        optimizer="adadelta",
        loss="mean_squared_error",
    ):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.hidden_units = hidden_units
        self.filters = filters

        self.optimizer = optimizer
        self.loss = loss
        self.history = None

        self.model, self.encoder, self.decoder = None, None, None

        if isConv:
            self.model, self.encoder, self.decoder = self.autoencoders_conv()
        else:
            self.model, self.encoder, self.decoder = self.autoencoders_flat()

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def summary(self, graph=False):
        self.model.summary()
        if graph:
            keras.utils.plot_model(
                self.model, to_file="autoencoder.png", show_shapes=True
            )

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def train(self, train_data, train_label, epochs=10, batch_size=1):
        self.history = self.model.fit(
            train_data,
            train_label,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=True,
        )

    def plot_loss(self):
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()

    def predict(self, test_data):
        return self.model.predict(test_data)

    def autoencoders_flat(self):
        input_layer = Input(shape=(self.input_shape,))

        for i in range(len(self.hidden_units)):
            if i == 0:
                layer = Dense(self.hidden_units[i], activation="relu")(input_layer)
            else:
                layer = Dense(self.hidden_units[i], activation="relu")(layer)

        middle_layer = layer

        for i in range(1, len(self.hidden_units)):
            if i == 1:
                layer = Dense(
                    self.hidden_units[len(self.hidden_units) - i - 1], activation="relu"
                )(middle_layer)
            else:
                layer = Dense(
                    self.hidden_units[len(self.hidden_units) - i - 1], activation="relu"
                )(layer)

        decoded = Dense(self.output_shape, activation="sigmoid")(layer)

        ae_model = Model(inputs=input_layer, outputs=decoded)
        encoder_model = Model(inputs=input_layer, outputs=middle_layer)
        decoder_model = Model(inputs=middle_layer, outputs=decoded)

        return ae_model, encoder_model, decoder_model

    def autoencoders_conv(self):
        input_layer = Input(shape=self.input_shape)

        for i in range(len(self.filters)):
            if i == 0:
                layer = Conv2D(
                    self.filters[i], (3, 3), activation="relu", padding="same"
                )(input_layer)
                layer = MaxPooling2D((2, 2), padding="same")(layer)
            else:
                layer = Conv2D(
                    self.filters[i], (3, 3), activation="relu", padding="same"
                )(layer)
                layer = MaxPooling2D((2, 2), padding="same")(layer)

        middle_layer = layer

        for i in range(len(self.filters)):
            layer = UpSampling2D((2, 2))(layer)
            layer = Conv2D(
                self.filters[len(self.filters) - i - 1],
                (3, 3),
                activation="relu",
                padding="same",
            )(layer)

        decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(layer)

        ae_model = Model(inputs=input_layer, outputs=decoded)
        encoder_model = Model(inputs=input_layer, outputs=middle_layer)
        decoder_model = Model(inputs=middle_layer, outputs=decoded)

        return ae_model, encoder_model, decoder_model


class baby_unet:
    def __init__(
        self,
        input_shape,
        output_shape,
        filters,
        optimizer="adadelta",
        loss="mean_squared_error",
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters

        self.optimizer = optimizer
        self.loss = loss
        self.history = None

        self.model = self.build_model(self.filters)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def summary(self, graph=False):
        self.model.summary()
        if graph:
            keras.utils.plot_model(
                self.model, to_file="baby_unet.png", show_shapes=True
            )

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def train(self, train_data, train_label, epochs=10, batch_size=1):
        self.history = self.model.fit(
            train_data,
            train_label,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=True,
        )

    def plot_loss(self):
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()

    def predict(self, test_data):
        return self.model.predict(test_data)

    def build_model(self, filters):
        input_layer = Input(shape=self.input_shape, name="input")

        # Contractive path
        for i in range(len(filters) - 1):
            name = "conv_up" + "_1" + "--" + str(i + 1)
            if i == 0:
                conv_1_down = Conv2D(
                    filters[i], 3, activation="relu", padding="same", name=name
                )(input_layer)
            else:
                conv_1_down = Conv2D(
                    filters[i], 3, activation="relu", padding="same", name=name
                )(pool)

            name = "conv_up" + "_2" + "--" + str(i + 1)
            conv_2_down = Conv2D(
                filters[i], 3, activation="relu", padding="same", name=name
            )(conv_1_down)
            name = "pool" + "--" + str(i + 1)
            pool = MaxPooling2D(pool_size=(2, 2), name=name)(conv_2_down)

        # Bottleneck
        name = "conv_b_1"
        conv_b = Conv2D(
            filters[len(filters) - 1], 3, activation="relu", padding="same", name=name
        )(pool)
        name = "conv_b_2"
        conv_b = Conv2D(
            filters[len(filters) - 1], 3, activation="relu", padding="same", name=name
        )(conv_b)

        # Expansive path
        for i in range(1, len(filters)):
            name = "up" + "--" + str(i + 1)
            if i == 1:
                up = UpSampling2D(size=(2, 2), name=name)(conv_b)
            else:
                up = UpSampling2D(size=(2, 2), name=name)(conv_2_up)

            name = "conv_down" + "_1" + "--" + str(i + 1)
            conv_1_up = Conv2D(
                filters[len(filters) - i - 1],
                3,
                activation="relu",
                padding="same",
                name=name,
            )(up)
            name = "conv_down" + "_2" + "--" + str(i + 1)
            conv_2_up = Conv2D(
                filters[len(filters) - i - 1],
                3,
                activation="relu",
                padding="same",
                name=name,
            )(conv_1_up)

        # Output
        name = "output"
        output_layer = Conv2D(
            self.output_shape[2], 3, activation="sigmoid", padding="same", name=name
        )(conv_2_up)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model


""" # Contractive path
input_layer = Input(shape=self.input_shape)
conv1 = Conv2D(64, 3, activation="relu", padding="same")(input_layer)
conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# Bottleneck

conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)

# Expansive path
up4 = UpSampling2D(size=(2, 2))(conv3)
conv4 = Conv2D(128, 3, activation="relu", padding="same")(up4)
conv4 = Conv2D(128, 3, activation="relu", padding="same")(conv4)

up5 = UpSampling2D(size=(2, 2))(conv4)
conv5 = Conv2D(64, 3, activation="relu", padding="same")(up5)
conv5 = Conv2D(64, 3, activation="relu", padding="same")(conv5)

# Output
output_layer = Conv2D(1, 1, activation="sigmoid")(conv5)

model = Model(inputs=input_layer, outputs=output_layer)

return model """
