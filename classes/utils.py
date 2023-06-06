import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import datetime
import os
import pandas as pd
import numpy as np
from scipy.ndimage import rotate

def plot_map(map):
    plt.imshow(map, cmap='jet')
    plt.colorbar()
    plt.show()

def plot_radial_profile(maps, azimut):
    for map in maps:
        angle = azimut + 90
        map_rot = rotate(map, angle, reshape=False)
        map_rot = np.abs(map_rot)
        map_rot[map_rot == 0] = np.min(map_rot[map_rot != 0])
        radial_profile = map_rot[map_rot.shape[0]//2, :]
        plt.plot(radial_profile)
    plt.show()

class ModelTemplate:
    def __init__(self, input_shape, output_shape, optimizer, loss):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        self.loss = loss
        self.history = None

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
    
    def train(self, train_data, train_label, epochs=10, batch_size=1):
        self.batch_size = batch_size
        # record history, allowing for consecutive training
        if self.history is None:
            self.history = self.model.fit(
                train_data,
                train_label,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                shuffle=True,
                verbose=1,
            )
        else:
            self.history = self.model.fit(
                train_data,
                train_label,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                shuffle=True,
                verbose=1,
                initial_epoch=self.history.epoch[-1],
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
        return self.model.predict(test_data, verbose=0)

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