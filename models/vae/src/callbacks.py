import tensorflow as tf
import numpy as np
import io
from keras.callbacks import Callback, TensorBoard
from PIL import Image
from .utils import normalize_dim

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

    def __init__(self, valid_data, **kwargs):
        super(TensorBoardExt, self).__init__(**kwargs)
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs={}):

        n_outputs = 3

        # generate images on validation data for tensorboard
        [xr, b], [yr, _] = next(self.valid_data)

        yf, flow = self.model.predict([xr, b])
        
        nbits = b.shape[1]
        max_delta = 2**nbits - 1
    
        d = np.ones_like(xr)[:, :, :, :8, :]

        # delta indicator
        for i in range(n_outputs):
            ratio = np.packbits(b[i:i+1, ::-1], axis=1) >> (8 - nbits)
            ratio = int(ratio / max_delta * d.shape[1])
            d[i, :ratio, ...] = 1
            d[i, ratio:, ...] = -0.2

        # scans
        channels = [xr, yr, yf, d]
        img0 = np.concatenate(channels, axis=3)[:3]
        img0 = np.concatenate(img0, axis=0)[:, 40, :, :]

        # flow
        flow_mean = flow[..., :3]
        logs['flow_mean'] = np.abs(flow_mean).mean()
        flow_mean = normalize_dim(flow_mean, axis=0)

        flow_std = flow[..., 3:]
        flow_std = np.exp(flow_std)
        logs['flow_std'] = np.abs(flow_std).mean()
        flow_std = normalize_dim(flow_std, axis=0)

        channels = [xr, flow_mean, flow_std]
        img1 = np.concatenate(channels, axis=4)
        img1 = np.moveaxis(img1, 4, 0)
        img1 = np.concatenate(img1, axis=3)[:3]
        img1 = np.concatenate(img1, axis=0)[:, 40, :, None]

        img_logs = {
            'xr_yr_yf': img0,
            'flow': img1
        }

        super(TensorBoardExt, self).on_epoch_end(epoch, logs, img_logs)
