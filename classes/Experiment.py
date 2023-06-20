import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import datetime
import os
import pandas as pd

from .UNet import UNet
from .AE import Autoencoder


class Experiment:
    def __init__(
        self,
        model_name,
        name,
        optimizer,
        learning_rate,
        loss,
        input_shape,
        output_shape,
        filters,
        blocks,
        dropout,
        epochs,
        batch_size,
        csv_path,
    ):
        self.model_name = model_name
        self.name = name
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters
        self.blocks = blocks
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None

        if isinstance(self.loss, list):
            loss = self.loss[0]
        else:
            loss = self.loss
        if model_name == "UNet":
            self.model = UNet(
                input_shape, output_shape, filters, blocks, dropout, optimizer, loss
            )
        elif model_name == "AE":
            self.model = Autoencoder(input_shape, optimizer, loss)
        else:
            raise Exception("Model name is not defined")

        self.csv_path = csv_path

        self.test_ssim = None
        self.test_psnr = None
        self.test_mse = None

    def make(self, train_data, train_label, test_data, test_label, save_model=False):
        # handle multi loss training
        if isinstance(self.loss, list):
            for loss, epoch in zip(self.loss, self.epochs):
                self.model.model.compile(optimizer=self.optimizer, loss=loss)
                self.model.train(train_data, train_label, epoch, self.batch_size)
        else:
            self.model.train(train_data, train_label, self.epochs, self.batch_size)
        model_save_path = os.path.join("models", self.name)
        if save_model:
            self.model.save_model(model_save_path)
        self.test_ssim = self.model.evaluate(test_data, test_label, metric="ssim")
        self.test_psnr = self.model.evaluate(test_data, test_label, metric="psnr")
        self.test_mse = self.model.evaluate(test_data, test_label, metric="mse")

    def save_experiment_csv(self):
        try:
            # Check if the CSV file exists and create it if it doesn't
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            "Experiment number",
                            "Name",
                            "Date",
                            "Optimizer",
                            "Learning rate",
                            "Loss",
                            "Input shape",
                            "Output shape",
                            "Filters",
                            "Dropout",
                            "Epochs",
                            "Batch size",
                            "Test SSIM",
                            "Test PSNR",
                            "Test MSE",
                        ]
                    )

            # Read the last experiment number from the CSV file
            last_experiment_number = 0
            with open(self.csv_path, "r") as file:
                number = 0
                if len(pd.read_csv(self.csv_path)) == 0:
                    number = 1
                else:
                    last_experiment_number = pd.read_csv(self.csv_path).iloc[-1][
                        "Experiment number"
                    ]
                    number = int(last_experiment_number) + 1
                file.close()

            optimizer_name = self.optimizer.__class__.__name__
            loss_name = self.loss.__class__.__name__

            # Write experiment results to the CSV file
            with open(self.csv_path, "a+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        number,
                        self.name,
                        datetime.datetime.now(),
                        optimizer_name,
                        self.learning_rate,
                        loss_name,
                        self.input_shape,
                        self.output_shape,
                        self.filters,
                        self.dropout,
                        self.epochs,
                        self.batch_size,
                        self.test_ssim,
                        self.test_psnr,
                        self.test_mse,
                    ]
                )
                file.close()

        except Exception as e:
            print(f"An error occurred while saving experiment results: {str(e)}")

    def get_model(self):
        return self.model
