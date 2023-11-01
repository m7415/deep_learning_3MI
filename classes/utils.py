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

def apply_mask(map, mask):
    return np.multiply(map, mask)

def plot_radial_profile(maps, azimut):
    for map in maps:
        angle = azimut + 90
        map_rot = rotate(map, angle, reshape=False, order=0)
        #map_rot = np.abs(map_rot)
        radial_profile = map_rot[map_rot.shape[0]//2, :]
        plt.plot(radial_profile)
    plt.show()

def get_center(map):
        max_sum = 0
        nom_i = 0
        nom_j = 0
        max = np.max(map)
        coord = np.where(map == max)
        i_m = coord[0][0]
        j_m = coord[1][0]
        for i in range(i_m - 2, i_m + 2):
            for j in range(j_m - 2, j_m + 2):
                if i < 0 or j < 0 or i >= 512 or j >= 512:
                    continue
                sum = np.sum(map[i : i + 2, j : j + 2])
                if sum > max_sum:
                    max_sum = sum
                    nom_i = i
                    nom_j = j
        return (nom_i, nom_j)

class ModelTemplate:
    def __init__(self, input_shape, output_shape, optimizer, loss):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        self.loss = loss
        self.history = None
    
    def save_model(self, model_path):
        tf.keras.saving.save_model(self.model, model_path, overwrite=True, save_format="tf")
    
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