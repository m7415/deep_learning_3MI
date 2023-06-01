import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.utils import vis_utils
from keras.utils.vis_utils import model_to_dot
import pydot
from keras.utils import plot_model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.optimizers import Adadelta
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
import csv
import datetime
import os
import pandas as pd

class ModelPackage:
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.history = None

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def train(self, train_data, train_label, epochs=10, batch_size=1):
        self.batch_size = batch_size
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
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.show()

    def predict(self, test_data):
        return self.model.predict(test_data)

# define the UNet class, which inherits from the Model class

class UNet(ModelPackage):
    def __init__(
        self,
        input_shape,
        output_shape,
        filters,
        dropout=0.2,
        optimizer="adam",
        loss="mean_squared_error",
    ):
        super().__init__(optimizer, loss)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters
        self.dropout = dropout

        self.model = self.build_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
    
    def summary(self, graph=False, graph_name=None):
        if not graph:
            self.model.summary()
        else:
            dot = pydot.Dot(graph_type="digraph", rankdir="UD")
            subdot = pydot.Subgraph(rank="same")
            node_dict = {}
            nodes_to_add = []
            contractive_path_names = ["input"]
            expansive_path_names = ["output"]

            for i in range(len(self.filters)):
                contractive_path_names.append(f"conv_up_1_{i+1}")
                contractive_path_names.append(f"conv_up_2_{i+1}")
                contractive_path_names.append(f"pool_{i+1}")
                expansive_path_names.append(f"conv_down_1_{i+1}")
                expansive_path_names.append(f"conv_down_2_{i+1}")
                expansive_path_names.append(f"up_{i+1}")

            for i, layer in enumerate(self.model.layers):
                if not isinstance(layer, tf.keras.layers.Concatenate):
                    node = pydot.Node(
                        layer.name,
                        label=f"{layer.name}\n-----------------\n{layer.output_shape}",
                        shape="box",
                        style="rounded, filled",
                    )
                    # save the index of the subgraph nodes
                    if layer.name == "conv_b_1":
                        nodes_to_add.append(i)
                        nodes_to_add.append(i - 1)
                    elif layer.name == "conv_b_2":
                        nodes_to_add.append(i)
                        nodes_to_add.append(i + 1)
                    # color the nodes
                    if layer.name in contractive_path_names:
                        node.set_fillcolor("pink")
                    elif layer.name == "conv_b_1" or layer.name == "conv_b_2":
                        node.set_fillcolor("lightgreen")
                    elif layer.name in expansive_path_names:
                        node.set_fillcolor("lightblue")
                    dot.add_node(node)
                    node_dict[layer.name] = node

            for i in nodes_to_add:
                subdot.add_node(node_dict[self.model.layers[i].name])

            dot.add_subgraph(subdot)

            # Connect the layers in the contractive path
            input_layer = self.model.get_layer("input")

            for i in range(len(self.filters) - 1):
                conv_layer_1 = self.model.get_layer(f"conv_up_1_{i+1}")
                if i == 0:
                    edge = pydot.Edge(
                        node_dict[input_layer.name], node_dict[conv_layer_1.name]
                    )
                else:
                    edge = pydot.Edge(
                        node_dict[pool_layer.name], node_dict[conv_layer_1.name]
                    )
                dot.add_edge(edge)
                conv_layer_2 = self.model.get_layer(f"conv_up_2_{i+1}")
                dot.add_edge(
                    pydot.Edge(
                        node_dict[conv_layer_1.name], node_dict[conv_layer_2.name]
                    )
                )
                pool_layer = self.model.get_layer(f"pool_{i+1}")
                dot.add_edge(
                    pydot.Edge(node_dict[conv_layer_2.name], node_dict[pool_layer.name])
                )

            # Connect the layers in the bottleneck
            conv_b_layer_1 = self.model.get_layer("conv_b_1")
            dot.add_edge(
                pydot.Edge(
                    node_dict[pool_layer.name],
                    node_dict[conv_b_layer_1.name],
                    constraint="false",
                )
            )
            conv_b_layer_2 = self.model.get_layer("conv_b_2")
            dot.add_edge(
                pydot.Edge(
                    node_dict[conv_b_layer_1.name],
                    node_dict[conv_b_layer_2.name],
                    constraint="false",
                )
            )

            # Connect the layers in the expansive path
            for i in range(1, len(self.filters)):
                up_layer = self.model.get_layer(f"up_{i}")
                if i == 1:
                    edge = pydot.Edge(
                        node_dict[conv_b_layer_2.name],
                        node_dict[up_layer.name],
                        constraint="false",
                    )
                else:
                    edge = pydot.Edge(
                        node_dict[up_layer.name],
                        node_dict[conv_layer_2.name],
                        dir="back",
                    )
                dot.add_edge(edge)
                conv_layer_1 = self.model.get_layer(f"conv_down_1_{i}")
                dot.add_edge(
                    pydot.Edge(
                        node_dict[conv_layer_1.name],
                        node_dict[up_layer.name],
                        dir="back",
                    )
                )
                conv_layer_2 = self.model.get_layer(f"conv_down_2_{i}")
                dot.add_edge(
                    pydot.Edge(
                        node_dict[conv_layer_2.name],
                        node_dict[conv_layer_1.name],
                        dir="back",
                    )
                )
            output_layer = self.model.get_layer("output")
            dot.add_edge(
                pydot.Edge(
                    node_dict[output_layer.name],
                    node_dict[conv_layer_2.name],
                    dir="back",
                )
            )

            # add the skip connections
            for i in range(len(self.filters) - 1):
                start_layer = self.model.get_layer(f"conv_up_2_{i+1}")
                end_layer = self.model.get_layer(f"conv_down_1_{len(self.filters)-i-1}")
                dot.add_edge(
                    pydot.Edge(
                        node_dict[start_layer.name],
                        node_dict[end_layer.name],
                        constraint="false",
                    )
                )

            # Save the graph
            graph_path = f"{graph_name}.png"
            print(f"U-shaped graph saved at: {graph_path}")

    def build_model(self):

        input_layer = Input(shape=self.input_shape, name="input")
        conv_layers_down = []
        pool = None

        # Contractive path
        for i, filters in enumerate(self.filters[:-1]):
            conv1_name = f"conv_down_1_{i+1}"
            conv2_name = f"conv_down_2_{i+1}"
            if i == 0:
                conv1 = Conv2D(
                    filters, 3, activation="relu", padding="same", name=conv1_name,
                )(input_layer)
                drop = Dropout(self.dropout)(conv1)
            else:
                conv1 = Conv2D(
                    filters, 3, activation="relu", padding="same", name=conv1_name,
                )(pool)
                drop = Dropout(self.dropout)(conv1)

            conv2 = Conv2D(
                filters, 3, activation="relu", padding="same", name=conv2_name
            )(drop)
            pool_name = f"pool_{i+1}"
            pool = MaxPooling2D(pool_size=(2, 2), name=pool_name)(conv2)
            conv_layers_down.append((conv1, conv2))

        # Bottleneck
        conv_b_1 = Conv2D(
            self.filters[-1], 3, activation="relu", padding="same", name="conv_b_1",
        )(pool)
        drop = Dropout(self.dropout)(conv_b_1)
        conv_b_2 = Conv2D(
            self.filters[-1], 3, activation="relu", padding="same", name="conv_b_2"
        )(drop)

        # Expansive path
        conv_layers_down = conv_layers_down[::-1]  # Reverse the list
        conv_layers_up = []
        up = None

        for i, filters in enumerate(self.filters[:-1][::-1]):
            conv1_name = f"conv_up_1_{i+1}"
            conv2_name = f"conv_up_2_{i+1}"
            up_name = f"up_{i+1}"
            if i == 0:
                up = UpSampling2D(size=(2, 2), name=up_name)(conv_b_2)
                skip_connection = conv_layers_down[i][
                    1
                ]  # Use the corresponding conv2 layer from the contractive path
                merge = concatenate([up, skip_connection], axis=-1)
            else:
                up = UpSampling2D(size=(2, 2), name=up_name)(conv2_up)
                skip_connection = conv_layers_down[i][
                    1
                ]  # Use the corresponding conv2 layer from the contractive path
                merge = concatenate([up, skip_connection], axis=-1)

            conv1_up = Conv2D(
                filters, 3, activation="relu", padding="same", name=conv1_name,
            )(merge)
            drop = Dropout(self.dropout)(conv1_up)
            conv2_up = Conv2D(
                filters, 3, activation="relu", padding="same", name=conv2_name
            )(drop)
            conv_layers_up.append((conv1_up, conv2_up))

        # Output
        output_layer = Conv2D(
            self.output_shape[2], 3, activation="sigmoid", padding="same", name="output"
        )(conv2_up)

        # Build model
        model = Model(inputs=input_layer, outputs=output_layer)

        return model
    
    def save_experiment_csv(self, test_data, test_label, csv_path, name):
        # if  csv_path does not exist, create it and write the header
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Experiment number", "Name", "Date", "Optimizer", "Loss", "Input shape", "Output shape", "Filters", "Dropout", "Epochs", "Batch size", "Test SSIM", "Test PSNR", "Test MSE"])
            # close the file
            file.close()
        with open(csv_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            reader = csv.reader(file)
            last_experiment_number = 0
            for row in reader:
                last_experiment_number = row[0]
            optimizer_name = self.optimizer.__class__.__name__
            loss_name = self.loss.__class__.__name__
            writer.writerow([int(last_experiment_number) + 1, name, datetime.datetime.now(), optimizer_name, loss_name, self.input_shape, self.output_shape, self.filters, self.dropout, self.history.params['epochs'], self.batch_size, self.evaluate(test_data, test_label, metric='ssim'), self.evaluate(test_data, test_label, metric='psnr'), self.evaluate(test_data, test_label, metric='mse')])    
    
    def print_experiment_csv(self, csv_path):
        df = pd.read_csv('unet.csv')
        df.head(len(df))

    def evaluate(self, test_data, test_label, metric):
        pred_label = self.predict(test_data)
        pred_label = pred_label.reshape(test_label.shape[0], test_label.shape[1], test_label.shape[2])
        # convert to double tensor
        pred_label = tf.convert_to_tensor(pred_label, dtype=tf.float32)
        test_label = tf.convert_to_tensor(test_label, dtype=tf.float32)
        if metric == "ssim":
            return tf.reduce_mean(
                tf.image.ssim(
                    test_label, pred_label, max_val=1.0, filter_size=11
                )
            ).numpy()
        elif metric == "psnr":
            return tf.reduce_mean(
                tf.image.psnr(test_label, pred_label, max_val=1.0)
            ).numpy()
        elif metric == "mse":
            return tf.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    test_label, pred_label
                )
            ).numpy()
        else:
            raise ValueError("Invalid metric")
        


class Autoencoder(ModelPackage):
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

        super().__init__(optimizer, loss)

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
