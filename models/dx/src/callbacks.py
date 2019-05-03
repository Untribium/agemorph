import tensorflow as tf
import numpy as np
from keras.callbacks import Callback, TensorBoard


class PredictionCallback(Callback):
    
    def __init__(self, data, update_freq=1, n_outputs=3):
        self.data = data
        self.update_freq = update_freq
        self.n_outputs = n_outputs

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.update_freq == 0:

            inputs, true = next(self.data)

            pred = self.model.predict(inputs, batch_size=inputs[0].shape[0])

            if not self.n_outputs:
                self.n_outputs = pred.shape[0]

            pred *= 100

            concat = np.concatenate([true[0], pred], axis=1)[:self.n_outputs]

            print(np.round(concat, decimals=2))


