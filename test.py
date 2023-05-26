from models import baby_unet, Autoencoder
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta

# Model
input_size = (512, 512, 3)
output_size = (512, 512, 1)
optimiser = Adadelta(learning_rate=1.0)
loss = tf.keras.losses.MeanSquaredError()
batch_size = 16
epochs = 5

unet = baby_unet(input_size, output_size, [16, 32, 64], optimiser, loss)

unet.summary(graph=True)

AE = Autoencoder(
    input_size, isConv=True, filters=[18, 8], optimizer=optimiser, loss=loss
)

AE.summary(graph=True)
