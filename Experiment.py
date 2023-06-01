import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import datetime
import os
import pandas as pd

from UNet import UNet
from AE import Autoencoder

class Experiment:
    def __init__(self, model_name, name, optimizer, loss, input_shape, output_shape, filters, dropout, epochs, batch_size):
        self.model_name = model_name
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None

        if model_name == "UNet":
            self.model = UNet(input_shape, output_shape, filters, dropout, optimizer, loss)
        elif model_name == "AE":
            self.model = Autoencoder(input_shape, optimizer=optimizer, loss=loss)
        else:
            raise Exception("Model name is not defined")
        
        self.csv_path = os.path.join("results", self.model_name + ".csv")
        
    def make(self, train_data, train_label, test_data, test_label, save_model=False):
        self.model.train(train_data, train_label, self.epochs, self.batch_size)
        model_save_path = os.path.join("models", self.name)
        if save_model:
            self.model.save_model(model_save_path)
        self.model.save_experiment_csv(test_data, test_label, self.csv_path, self.name)
    
    def get_model(self):
        return self.model