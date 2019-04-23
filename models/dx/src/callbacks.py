import tensorflow as tf
import numpy as np
from keras.callbacks import Callback, TensorBoard


class PredictionCallback(Callback):
    
    def __init__(self, data, update_freq=1):
        self.data = data
        self.update_freq = update_freq

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.update_freq == 0:
            X, y = next(self.data)
            logits = self.model.predict(X, batch_size=X[0].shape[0])
            print(logits)
            print(y)


