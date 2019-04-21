import tensorflow as tf
import numpy as np
import io
from keras.callbacks import Callback, TensorBoard
from PIL import Image


def convert_delta(gen, max_delta, int_steps, kl_dummy):
    
    while True:
        
        imgs, lbls = next(gen)

        delta_norm = lbls[0] / max_delta

        # binary representation
        scaled = (delta_norm * 255).astype(np.uint8).reshape((-1, 1))
        bits = np.unpackbits(scaled, axis=1)[:, int_steps::-1]

        yield [imgs[0], bits, *lbls[1:]], [imgs[1], kl_dummy]


def normalize_dim(tensor, axis):
    ndims = len(tensor.shape)

    axes = np.arange(ndims)
    axes = np.delete(axes, axis)

    maxs = np.abs(tensor).max(axis=tuple(axes))
    maxs[maxs == 0] = 1 # avoid div by 0

    slicer = [None] * ndims
    slicer[axis] = slice(None)

    tensor /= maxs[tuple(slicer)]

    return tensor


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


class TensorBoardVAE(TensorBoardImage):

    def __init__(self, valid_data, **kwargs):
        super(TensorBoardVAE, self).__init__(**kwargs)
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs={}):

        [xr, d], [yr, _] = next(self.valid_data)

        yf, flow = self.model.predict([xr, d])

        # scans
        channels = [xr, yr, yf]
        img0 = np.concatenate(channels, axis=3)[:3]
        img0 = np.concatenate(img0, axis=0)[:, 40, :, :]

        # flow
        flow_mean = flow[..., :3]
        flow_mean = normalize_dim(flow_mean, axis=0)

        flow_std = flow[..., 3:]
        flow_std = np.exp(flow_std)
        flow_std = normalize_dim(flow_std, axis=0)

        channels = [flow_mean, flow_std]
        img1 = np.concatenate(channels, axis=4)
        img1 = np.moveaxis(img1, 4, 0)
        img1 = np.concatenate(img1, axis=3)[:3]
        img1 = np.concatenate(img1, axis=0)[:, 40, :, None]

        img_logs = {
            'xr_yr_yf': img0,
            'flow': img1
        }

        super(TensorBoardVAE, self).on_epoch_end(epoch, logs, img_logs)


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


