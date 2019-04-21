import tensorflow as tf
import numpy as np
import io
from keras.callbacks import Callback, TensorBoard
from PIL import Image


def make_image(tensor):
    h, w, c = tensor.shape

    image = Image.fromarray(tensor.squeeze())
    
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=image_string)


class TensorBoardImage(TensorBoard):
    
    def __init__(self, **kwargs):
        super(TensorBoardImage, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}, img_logs={}):

        for key, img in img_logs.items():

            img_int = (img * 127 + 127).astype('uint8')

            image = make_image(img_int)
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, image=image)])
            self.writer.add_summary(summary, epoch)

        self.writer.flush()

        super(TensorBoardImage, self).on_epoch_end(epoch, logs)


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


