import tensorflow as tf
import numpy as np
import io
from keras.callbacks import Callback, TensorBoard
from PIL import Image
from .utils import normalize_dim, to_int


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


class TensorBoardExt(TensorBoardImage):

    def __init__(self, valid_data, int_steps, **kwargs):
        super(TensorBoardExt, self).__init__(**kwargs)
        self.valid_data = valid_data
        self.int_steps = int_steps

    def on_epoch_end(self, epoch, logs={}):

        n_outputs = 3

        # generate images on validation data for tensorboard
        inputs, labels, batch = next(self.valid_data)

        yf, flow_params = self.model.predict(inputs)

        xr, _ = inputs
        yr, _ = labels

        delta = batch['delta_t']

        # delta bin to int to channel        
        delta = delta.reshape((-1, 1, 1, 1, 1))
        delta = np.ones_like(xr, dtype=np.float32) * delta

        # delta indicator
        lin = np.linspace(0, 1, xr.shape[1])
        deltas = np.ones_like(xr) * lin[None, :, None, None, None]
        deltas = ((deltas < delta) * 1.2 - 0.2)[:, :, :, :10, :]

        # scans
        slice_scans = xr.shape[2] // 2
        channels = [xr, yr, yf, deltas]
        img0 = np.concatenate(channels, axis=3)[:n_outputs]
        img0 = np.concatenate(img0, axis=0)[:, slice_scans, :, :]

        # flow
        slice_flows = flow_params.shape[2] // 2
        flow_mean = flow_params[..., :3]
        logs['flow_mean'] = np.abs(flow_mean).mean()
        flow_mean = normalize_dim(flow_mean, axis=0)

        flow_std = flow_params[..., 3:]
        flow_std = np.exp(flow_std)
        logs['flow_std'] = np.abs(flow_std).mean()
        flow_std = normalize_dim(flow_std, axis=0)

        channels = [flow_mean, flow_std]
        img1 = np.concatenate(channels, axis=4)
        img1 = np.moveaxis(img1, 4, 0)
        img1 = np.concatenate(img1, axis=3)[:n_outputs]
        img1 = np.concatenate(img1, axis=0)[:, slice_flows, :, None]

        img_logs = {
            'scans': img0,
            'flows': img1
        }

        super(TensorBoardExt, self).on_epoch_end(epoch, logs, img_logs)
