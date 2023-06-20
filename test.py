import tensorflow as tf

from classes.data_wraper import Dataset
from classes.UNet import UNet
from classes.Experiment import Experiment
from classes.utils import plot_map, plot_radial_profile, apply_mask


# Model
input_size = (512, 512, 3)
output_size = (512, 512, 1)
optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()
batch_size = 16
epochs = 5

unet = UNet(input_size, output_size, [32, 64, 128, 256], 0.2, optimiser, loss)

unet.summary()
