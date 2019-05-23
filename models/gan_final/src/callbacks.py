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


def summaries(cri_logs, gen_logs, use_reg, use_clf):
        
    summaries = {
        'cri_loss_total':   cri_logs[0],
        'cri_loss_ws_real': cri_logs[1],
        'cri_loss_ws_fake': cri_logs[2],
        'cri_loss_gp':      cri_logs[3],
        'gen_loss_total':   gen_logs[0],
        'gen_loss_ws':      gen_logs[1],
        'gen_loss_l1':      gen_logs[2],
        'gen_loss_kl':      gen_logs[3],
        'gen_loss_sparse':  gen_logs[4]
    }

    i = 6 
 
    if use_reg:
        summaries['gen_loss_age'] = gen_logs[i]
        i += 1

    if use_clf:
        summaries['gen_loss_dx'] = gen_logs[i]

    return summaries


class TensorBoardExt(TensorBoard):
    
    def __init__(self, use_reg, use_clf, **kwargs):

        self.use_reg = use_reg
        self.use_clf = use_clf

        super(TensorBoardExt, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, cri_logs, gen_logs):

        logs = summaries(cri_logs, gen_logs, self.use_reg, self.use_clf)

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

    def __init__(self, cri_data, gen_data, cri_model, gen_model, freq, steps,
                                    use_reg, use_clf, n_outputs=3, **kwargs):

        self.cri_data = cri_data
        self.gen_data = gen_data
        self.freq = freq
        self.steps = steps
        self.cri_model = cri_model
        self.gen_model = gen_model

        self.use_reg = use_reg
        self.use_clf = use_clf

        self.n_outputs = n_outputs

        super(TensorBoardVal, self).__init__(**kwargs)

    
    def on_epoch_end(self, epoch):
 
        if epoch % self.freq != 0:
            return

        # --- scalar summaries ---
        cri_logs_valid = np.zeros((self.steps, len(self.cri_model.outputs)+1))
        gen_logs_valid = np.zeros((self.steps, len(self.gen_model.outputs)+1))

        # run steps
        for v_step in range(self.steps):

            inputs, labels, _ = next(self.cri_data)
            cri_logs_valid[v_step] = self.cri_model.test_on_batch(inputs, labels)
            
            inputs, labels, batch = next(self.gen_data)
            gen_logs_valid[v_step] = self.gen_model.test_on_batch(inputs, labels)
       
        # take mean 
        cri_logs_valid = cri_logs_valid.mean(axis=0).tolist()
        gen_logs_valid = gen_logs_valid.mean(axis=0).tolist()

        logs = summaries(cri_logs_valid, gen_logs_valid, self.use_reg, self.use_clf)

        # --- image summaries ---

        # predict y_hat and flow params
        gen_out = self.gen_model.predict(inputs)

        outputs = [
                batch['img_path_0'],    # x
                batch['img_path_1'],    # y
                batch['delta_t'],       # delta scaled
                gen_out[1],             # y_hat
                gen_out[2],             # flow_params
                gen_out[4]              # flow
        ]

        outputs = list(zip(*outputs))

        logs['gen_flow_mean'] = 0
        logs['gen_flow_std'] = 0

        def img_strip(xr, yr, delta, yf, flow_params, flow):

            # delta indicator
            lin = np.linspace(0, 1, xr.shape[0])
            ind = (lin < delta) * 1.2 - 0.2
            ind = ind[..., None].repeat(xr.shape[1], axis=-1)
            ind = ind[..., None, None].repeat(10, axis=-2)

            # scans
            scans = [xr, yr, yf, ind]
            scans = np.concatenate(scans, axis=2)

            def _normalize(tensor):
                max_v = np.abs(tensor).max()
                max_v = 1 if max_v == 0 else max_v
                tensor /= max_v
                return tensor
            
            # --- flows ---
            flow_mag = np.sqrt((flow * flow).sum(axis=-1))
            flow_mag = _normalize(flow_mag)
            flow_mag = flow_mag[..., None].repeat(3, axis=-1)
            
            flow = _normalize(flow)

            flow_dim = [flow[..., d:d+1].repeat(3, axis=-1) for d in range(3)]

            flows = [xr.repeat(3, axis=-1), flow_mag, flow, *flow_dim]
            flows = np.concatenate(flows, axis=2)

            # --- flows mean ---
            mean_rgb = flow_params[..., :3]

            # magnitude
            mean_mag = np.sqrt((mean_rgb * mean_rgb).sum(axis=-1))
            logs['gen_flow_mean'] += np.abs(mean_mag).mean()
            mean_mag = _normalize(mean_mag)
            mean_mag = mean_mag[..., None].repeat(3, axis=-1)

            mean_rgb = _normalize(mean_rgb)

            # repeat channel to match rgb shape
            mean_dim = [mean_rgb[..., d:d+1].repeat(3, axis=-1) for d in range(3)]

            flows_mean = [mean_mag, mean_rgb, *mean_dim]
            flows_mean = np.concatenate(flows_mean, axis=2)

            # --- flows_std ---
            std_rgb = flow_params[..., 3:]
            std_rgb = np.exp(std_rgb)

            # magnitude
            std_mag = np.sqrt((std_rgb * std_rgb).sum(axis=-1))
            logs['gen_flow_std'] += np.abs(std_mag).mean()
            std_mag = _normalize(std_mag)
            std_mag = std_mag[..., None].repeat(3, axis=-1)

            std_rgb = _normalize(std_rgb)

            # repeat channel to match rgb shape
            std_dim = [std_rgb[..., d:d+1].repeat(3, axis=-1) for d in range(3)]

            flows_std = [std_mag, std_rgb, *std_dim]
            flows_std = np.concatenate(flows_std, axis=2)

            return scans, flows, flows_mean, flows_std
         
          
        img_strips = [img_strip(*o) for o in outputs[:self.n_outputs]]
        img_strips = list(zip(*img_strips))

        logs['gen_flow_mean'] /= self.n_outputs 
        logs['gen_flow_std'] /= self.n_outputs 

        scans = np.concatenate(img_strips[0], axis=0)
        flows = np.concatenate(img_strips[1], axis=0)
        flows_mean = np.concatenate(img_strips[2], axis=0)
        flows_std = np.concatenate(img_strips[3], axis=0)

        sli_fs = batch['img_path_0'].shape[2] // 2  # full size
        sli_vr = gen_out[2].shape[2] // 2           # vel resized

        img_logs = {
            'scans': scans[:, sli_fs, :, :],
            'flows': flows[:, sli_fs, :, :],
            'flows_mean': flows_mean[:, sli_vr, :, :],
            'flows_std': flows_std[:, sli_vr, :, :]
        }
            
        super(TensorBoardVal, self).on_epoch_end(epoch, logs, img_logs) 
