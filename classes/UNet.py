import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from keras.regularizers import l2

import pydot

from .utils import ModelTemplate

class UNet(ModelTemplate):
    def __init__(
        self,
        input_shape = (512, 512, 1),
        output_shape = (512, 512, 1),
        filters = [64, 128, 256],
        block_sizes = [2, 2, 2],
        dropout=0.2,
        optimizer="adam",
        loss="mean_squared_error",
        model_path=None,
    ):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
            super().__init__(
                self.model.input_shape,
                self.model.output_shape,
                self.model.optimizer,
                self.model.loss,
            )
            print(f"Loaded model from {model_path}")
            return

        super().__init__(input_shape, output_shape, optimizer, loss)

        self.filters = filters
        self.block_sizes = block_sizes
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
                for j in range(self.block_sizes[0]):
                    contractive_path_names.append(f"conv_down_{j+1}_{i+1}")
                contractive_path_names.append(f"pool_{i+1}")
                for j in range(self.block_sizes[2]):
                    expansive_path_names.append(f"conv_up_{j+1}_{i+1}")
                expansive_path_names.append(f"up_{i+1}")

            for i, layer in enumerate(self.model.layers):
                if not isinstance(layer, tf.keras.layers.Concatenate) and not isinstance(
                    layer, tf.keras.layers.Dropout
                ):
                    node = pydot.Node(
                        layer.name,
                        label=f"{layer.name}\n-----------------\n{layer.output_shape}",
                        shape="box",
                        style="rounded, filled",
                    )
                    # save the index of the subgraph nodes
                    for j in range(self.block_sizes[1]):
                        if layer.name == f"conv_b_{j+1}":
                            nodes_to_add.append(i)
                    # color the nodes
                    if layer.name in contractive_path_names:
                        node.set_fillcolor("pink")
                    elif layer.name in expansive_path_names:
                        node.set_fillcolor("lightblue")
                    else:
                        node.set_fillcolor("lightgreen")
                    dot.add_node(node)
                    node_dict[layer.name] = node

            # add the node before the bottleneck and after the bottleneck
            nodes_to_add.append(np.min(nodes_to_add) - 1)
            nodes_to_add.append(np.max(nodes_to_add) + 1)
            for i in nodes_to_add:
                subdot.add_node(node_dict[self.model.layers[i].name])

            dot.add_subgraph(subdot)

            # Connect the layers in the contractive path
            input_layer = self.model.get_layer("input")

            for i in range(len(self.filters) - 1):
                for j in range(self.block_sizes[0]):
                    conv_layer = self.model.get_layer(f"conv_down_{j+1}_{i+1}")
                    if j == 0:
                        if i == 0:
                            edge = pydot.Edge(
                                node_dict[input_layer.name], node_dict[conv_layer.name]
                            )
                        else:
                            edge = pydot.Edge(
                                node_dict[pool_layer.name], node_dict[conv_layer.name]
                            )
                    else:
                        edge = pydot.Edge(
                            node_dict[conv_layer_2.name], node_dict[conv_layer.name]
                        )
                    dot.add_edge(edge)

                conv_layer_2 = self.model.get_layer(f"conv_down_2_{i+1}")
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
                conv_layer_1 = self.model.get_layer(f"conv_up_1_{i}")
                dot.add_edge(
                    pydot.Edge(
                        node_dict[conv_layer_1.name],
                        node_dict[up_layer.name],
                        dir="back",
                    )
                )
                conv_layer_2 = self.model.get_layer(f"conv_up_2_{i}")
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
                start_layer = self.model.get_layer(f"conv_down_2_{i+1}")
                end_layer = self.model.get_layer(f"conv_up_1_{len(self.filters)-i-1}")
                dot.add_edge(
                    pydot.Edge(
                        node_dict[start_layer.name],
                        node_dict[end_layer.name],
                        constraint="false",
                    )
                )

            # Save the graph
            graph_path = f"{graph_name}.png"
            dot.write_png(graph_path)
            print(f"U-shaped graph saved at: {graph_path}")

    def build_model(self):

        input_layer = Input(shape=self.input_shape, name="input")
        conv_layers_down = []
        pool = None

        w_decay = 0.00001
        #, kernel_regularizer=l2(w_decay), bias_regularizer=l2(w_decay)

        initializer = 'he_uniform'

        block_size = self.block_sizes[0]
        # Contractive path
        for i, filters in enumerate(self.filters[:-1]):
            # Convolutional block
            for j in range(block_size):
                conv_name = f"conv_down_{j+1}_{i+1}"
                if j == 0:
                    if i == 0:
                        conv_down = Conv2D(
                            filters, 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                        )(input_layer)
                    else:
                        conv_down = Conv2D(
                            filters, 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                        )(pool)
                else:
                    conv_down = Conv2D(
                        filters, 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                    )(drop)
                batch_norm = BatchNormalization()(conv_down)
                drop = Dropout(self.dropout)(batch_norm)

            pool_name = f"pool_{i+1}"
            pool = MaxPooling2D(pool_size=(2, 2), name=pool_name)(drop)
            conv_layers_down.append(conv_down)

        block_size = self.block_sizes[1]
        # Bottleneck
        for j in range(block_size):
            conv_name = f"conv_b_{j+1}"
            if j == 0:
                conv_b = Conv2D(
                    self.filters[-1], 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                )(pool)
            else:
                conv_b = Conv2D(
                    self.filters[-1], 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                )(drop)
            batch_norm = BatchNormalization()(conv_b)
            drop = Dropout(self.dropout)(batch_norm)

        conv_layers_down = conv_layers_down[::-1]  # Reverse the list

        block_size = self.block_sizes[2]
        # Expansive path
        for i, filters in enumerate(self.filters[:-1][::-1]):
            for j in range(block_size):
                conv_name = f"conv_up_{j+1}_{i+1}"
                up_name = f"up_{i+1}"
                if j == 0:
                    if i == 0:
                        up = UpSampling2D(size=(2, 2), name=up_name)(drop)
                    else:
                        up = UpSampling2D(size=(2, 2), name=up_name)(conv_up)
                    skip_connection = conv_layers_down[i]
                    merge = concatenate([up, skip_connection], axis=-1)
                    conv_up = Conv2D(
                        filters, 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                    )(merge)
                else:
                    conv_up = Conv2D(
                        filters, 3, activation="relu", kernel_initializer=initializer, padding="same", name=conv_name,
                    )(drop)
                batch_norm = BatchNormalization()(conv_up)
                drop = Dropout(self.dropout)(batch_norm)

        # Output
        output_layer = Conv2D(
            self.output_shape[2], 3, activation="sigmoid", kernel_initializer='glorot_uniform', padding="same", name="output"
        )(drop)

        # Build model
        model = Model(inputs=input_layer, outputs=output_layer)

        return model
    