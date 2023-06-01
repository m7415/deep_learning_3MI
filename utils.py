import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import datetime
import os
import pandas as pd

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
    
    def save_experiment_csv(self, test_data, test_label, csv_path, name):
        # if the results folder does not exist, create it
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Experiment number", "Name", "Date", "Optimizer", "Loss", "Input shape", "Output shape", "Filters", "Dropout", "Epochs", "Batch size", "Test SSIM", "Test PSNR", "Test MSE"])
            # close the file
            file.close()
        with open(csv_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            number = 0
            if len(pd.read_csv(csv_path)) == 0:
                number = 1
            else:
                last_experiment_number = pd.read_csv('unet.csv').iloc[-1]['Experiment number']
                number = int(last_experiment_number) + 1
            optimizer_name = self.optimizer.__class__.__name__
            loss_name = self.loss.__class__.__name__
            writer.writerow([number, name, datetime.datetime.now(), optimizer_name, loss_name, self.input_shape, self.output_shape, self.filters, self.dropout, self.history.params['epochs'], self.batch_size, self.evaluate(test_data, test_label, metric='ssim'), self.evaluate(test_data, test_label, metric='psnr'), self.evaluate(test_data, test_label, metric='mse')])    
        file.close()

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