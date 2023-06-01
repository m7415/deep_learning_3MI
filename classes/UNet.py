import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout

import pydot

from utils import ModelTemplate

class UNet(ModelTemplate):
    def __init__(
        self,
        input_shape,
        output_shape,
        filters,
        dropout=0.2,
        optimizer="adam",
        loss="mean_squared_error",
    ):
        super().__init__(input_shape, output_shape, optimizer, loss)

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
                contractive_path_names.append(f"conv_down_1_{i+1}")
                contractive_path_names.append(f"conv_down_2_{i+1}")
                contractive_path_names.append(f"pool_{i+1}")
                expansive_path_names.append(f"conv_up_1_{i+1}")
                expansive_path_names.append(f"conv_up_2_{i+1}")
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
                conv_layer_1 = self.model.get_layer(f"conv_down_1_{i+1}")
                if i == 0:
                    edge = pydot.Edge(
                        node_dict[input_layer.name], node_dict[conv_layer_1.name]
                    )
                else:
                    edge = pydot.Edge(
                        node_dict[pool_layer.name], node_dict[conv_layer_1.name]
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
    