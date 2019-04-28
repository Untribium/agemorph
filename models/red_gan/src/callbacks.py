import tensorflow as tf
import numpy as np
import io
from .utils import normalize
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


def summaries(cri_logs, gen_logs):
        
    summaries = {
        'cri_loss_total':   cri_logs[0],
        'cri_loss_ws_real': cri_logs[1],
        'cri_loss_ws_fake': cri_logs[2],
        'cri_loss_gp':      cri_logs[3],
        'gen_loss_total':   gen_logs[0],
        'gen_loss_ws':      gen_logs[1],
        'gen_loss_age':     gen_logs[2],
        'gen_loss_l1':      gen_logs[3],
        'gen_loss_kl':      gen_logs[4]
    }

    return summaries


class TensorBoardExt(TensorBoard):
    
    def __init__(self, **kwargs):
        super(TensorBoardExt, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, cri_logs, gen_logs):
        logs = summaries(cri_logs, gen_logs)
        super(TensorBoardExt, self).on_epoch_end(epoch, logs)


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


class TensorBoardVal(TensorBoardImage):

    def __init__(self, cri_model, gen_model, data, freq, steps, batch_size, kl_dummy, f_dummy, **kwargs):
        self.data = data
        self.freq = freq
        self.steps = steps
        self.gen_model = gen_model
        self.cri_model = cri_model

        self.kl_dummy = kl_dummy
        self.f_dummy = f_dummy

        # critic labels 
        self.real = np.ones((batch_size, 1)) * (-1) # real labels
        self.fake = np.ones((batch_size, 1))        # fake labels
        self.avgd = np.ones((batch_size, 1))        # dummy labels for gradient penalty
        self.zero = np.zeros((batch_size, 1))       # zero labels for age delta loss

        super(TensorBoardVal, self).__init__(**kwargs)

    
    def on_epoch_end(self, epoch):
 
        if epoch % self.freq != 0:
            return

        # scalar summaries
        cri_logs_valid = np.zeros((self.steps, len(self.cri_model.outputs)+1))
        gen_logs_valid = np.zeros((self.steps, len(self.gen_model.outputs)+1))

        # run steps
        for v_step in range(self.steps):

            imgs, lbls = next(self.data)

            cri_in = [imgs[0], imgs[1], lbls[1]]
            cri_true = [self.real, self.fake, self.avgd]
            cri_logs_valid[v_step] = self.cri_model.test_on_batch(cri_in, cri_true)
            
            gen_in = [imgs[0], lbls[1]]
            gen_true = [self.real, self.zero, imgs[0], self.kl_dummy, self.f_dummy]
            gen_logs_valid[v_step] = self.gen_model.test_on_batch(gen_in, gen_true)
       
        # take mean 
        cri_logs_valid = cri_logs_valid.mean(axis=0).tolist()
        gen_logs_valid = gen_logs_valid.mean(axis=0).tolist()

        logs = summaries(cri_logs_valid, gen_logs_valid)

        # image summaries
        num_outputs = 3

        # predict y_hat and flow params
        _, _, yf_gen, flow_gen, _ = self.gen_model.predict(gen_in)

        # delta indicator
        lin = np.linspace(0, 1, imgs[0].shape[1])
        ind = lbls[0][:, None] > lin[None, :]
        ind = ind * 1.2 - 0.2
        ind = ind[..., None].repeat(imgs[0].shape[2], axis=-1)
        ind = ind[..., None].repeat(10, axis=-1)
        
        # combine scans and delta
        channels = [imgs[0], imgs[1], yf_gen, ind]
        scans = np.concatenate(channels, axis=3)[:num_outputs]
        scans = np.concatenate(img, axis=0)

        # all flows
        flow_mean = flow_gen[..., :3]
        logs['gen_flow_mean'] = np.abs(flow_mean).mean()
        flow_mean = normalize(flow_mean, axis=0)

        flow_std = flow_gen[..., 3:]
        flow_std = np.exp(flow_std)
        logs['gen_flow_std'] = np.abs(flow_std).mean()
        flow_std = normalize(flow_std, axis=0)
       
        # rgb flows
        flow_mean_rgb = flow_mean[None, ...]
        flow_std_rgb = flow_std[None, ...]

        # extend grayscales to 3 channels so can be displayed alongside rgb
        flow_mean = np.moveaxis(flow_mean, -1, 0)
        flow_mean = flow_mean[..., None].repeat(3, axis=-1)
        
        flow_std = np.moveaxis(flow_std, -1, 0)
        flow_std = flow_std[..., None].repeat(3, axis=-1)

        # combine flows
        flows = [flow_mean_rgb, flow_mean, flow_std_rgb, flow_std]
        flows = np.concatenate(channels, axis=0)
        flows = np.concatenate(flows, axis=3)[:num_outputs]
        flows = np.concatenate(flows, axis=0)

        sli = imgs[0].shape[2] // 2

        img_logs = {
                'flows': flows[:, sli, :, :]
                'scans': scans[:, sli, :, :]
        }

        super(TensorBoardVal, self).on_epoch_end(epoch, logs, img_logs) 
