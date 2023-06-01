import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

from utils import ModelTemplate

class Autoencoder(ModelTemplate):
    def __init__(
        self,
        input_shape,
        isConv=False,
        hidden_units=None,
        filters=None,
        optimizer="adadelta",
        loss="mean_squared_error",
    ):
        super().__init__(input_shape, input_shape, optimizer, loss)

        self.hidden_units = hidden_units
        self.filters = filters

        self.model, self.encoder, self.decoder = self.build_autoencoders(isConv)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def summary(self, graph=False):
        self.model.summary()
        if graph:
            keras.utils.plot_model(
                self.model, to_file="autoencoder.png", show_shapes=True
            )

    def build_autoencoders(self, isConv):
        if isConv:
            return self.build_autoencoders_conv()
        else:
            return self.build_autoencoders_flat()

    def build_autoencoders_flat(self):
        input_layer = Input(shape=(self.input_shape,))

        for i, units in enumerate(self.hidden_units):
            if i == 0:
                layer = Dense(units, activation="relu")(input_layer)
            else:
                layer = Dense(units, activation="relu")(layer)

        middle_layer = layer

        for i, units in enumerate(self.hidden_units[::-1]):
            layer = Dense(units, activation="relu")(layer)

        decoded = Dense(self.output_shape, activation="sigmoid")(layer)

        ae_model = Model(inputs=input_layer, outputs=decoded)
        encoder_model = Model(inputs=input_layer, outputs=middle_layer)
        decoder_model = Model(inputs=middle_layer, outputs=decoded)

        return ae_model, encoder_model, decoder_model

    def build_autoencoders_conv(self):
        input_layer = Input(shape=self.input_shape)

        for i, filters in enumerate(self.filters):
            if i == 0:
                layer = Conv2D(filters, (3, 3), activation="relu", padding="same")(
                    input_layer
                )
                layer = MaxPooling2D((2, 2), padding="same")(layer)
            else:
                layer = Conv2D(filters, (3, 3), activation="relu", padding="same")(
                    layer
                )
                layer = MaxPooling2D((2, 2), padding="same")(layer)

        middle_layer = layer

        for i, filters in enumerate(self.filters[::-1]):
            layer = UpSampling2D((2, 2))(layer)
            layer = Conv2D(filters, (3, 3), activation="relu", padding="same")(layer)

        decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(layer)

        ae_model = Model(inputs=input_layer, outputs=decoded)
        encoder_model = Model(inputs=input_layer, outputs=middle_layer)
        decoder_model = Model(inputs=middle_layer, outputs=decoded)

        return ae_model, encoder_model, decoder_model