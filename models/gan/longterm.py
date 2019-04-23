# python
import os
import sys
from argparse import ArgumentParser

# third party
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import keras
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import nibabel as nib

# project
from src import networks, losses, datagenerators
from src.utils import convert_delta

sys.path.append('../../ext/neuron')

import neuron.layers as nrn_layers


def predict(gpu_id, img_path, out_dir, batch_size, gen_model_file, start, stop, step):

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # check generator model file exists 
    assert os.path.isfile(gen_model_file), "generator model file does not exist"

    gen_model_file = os.path.abspath(gen_model_file)

    # extract run directory and model checkpoint name
    model_dir, model_name = os.path.split(gen_model_file)
    model_name = os.path.splitext(model_name)[0]

    # load model config
    config_path = os.path.join(model_dir, 'config.pkl')
    
    assert os.path.isfile(config_path), "model_config not found"
    
    model_config = pickle.load(open(config_path, 'rb'))

    # load nifti
    assert os.path.isfile(img_path), "img file not found"
    img_path = os.path.abspath(img_path)

    nii = nib.load(img_path)
    vol = np.squeeze(nii.get_data().astype(np.float32))

    img_id = os.path.basename(img_path).split('.')[0]

    # create out_dir (run_dir/predict/model_name_delta)
    if out_dir is None or out_dir == '':
        out_dir = os.path.join(model_dir, 'predict', 'longterm', model_name)
   
    out_dir = os.path.join(out_dir, img_id)
 
    print('out_dir:', out_dir)
 
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # extract config variables
    vol_shape = model_config['vol_shape']
    max_delta = model_config['max_delta']
    vel_resize = model_config['vel_resize']
    prior_lambda = model_config['prior_lambda']
    cri_loss_weights = model_config['cri_loss_weights']
    gen_loss_weights = model_config['gen_loss_weights']
    enc_nf = model_config['enc_nf']
    dec_nf = model_config['dec_nf']
    cri_base_nf = model_config['cri_base_nf']
    int_steps = model_config['int_steps']
    ti_flow = model_config['ti_flow']

    flow_shape = tuple(int(d * vel_resize) for d in vol_shape)

    csv = pd.DataFrame()
    csv['delta_t'] = np.arange(start, stop, step) * 365
    csv['img_path'] = img_path
    csv['img_id'] = img_id

    # write meta to out_dir
    csv_out_path = os.path.join(out_dir, 'meta.csv')
    csv.to_csv(csv_out_path, index=False)

    img_keys = ['img_path']
    lbl_keys = ['delta_t', 'img_id', 'delta_t']
 
    # datagenerator (from meta in out_dir!) 
    test_csv_data = datagenerators.csv_gen(csv_out_path, img_keys=img_keys,
                                    lbl_keys=lbl_keys, batch_size=batch_size,
                                    split=None, n_epochs=1, sample=False,
                                    shuffle=False)

    test_data = convert_delta(test_csv_data, max_delta, int_steps)

    kl_dummy = np.zeros((batch_size, *flow_shape, len(vol_shape)-1))

    with tf.device(gpu):

        print('loading model')

        loss_class = losses.GANLosses(prior_lambda=prior_lambda,flow_shape=flow_shape)

        # create generator model
        _, gen_net = networks.gan_models(vol_shape, batch_size, loss_class,
                                         cri_loss_weights=cri_loss_weights,
                                         cri_optimizer=Adam(),
                                         gen_loss_weights=gen_loss_weights,
                                         gen_optimizer=Adam(),
                                         enc_nf=enc_nf, dec_nf=dec_nf,
                                         cri_base_nf=cri_base_nf,
                                         vel_resize=vel_resize,
                                         ti_flow=ti_flow,
                                         int_steps=int_steps)

        # load weights into model
        gen_net.load_weights(gen_model_file)

        print('starting predict')

        # predict
        for i, (imgs, lbls) in enumerate(test_data):

            if i % 10 == 0:
                print('step', i)

            # generate
            yf, flow, flow_ti, Df = gen_net.predict([imgs[0], lbls[0], lbls[1]])

            img_ids = lbls[2]
            deltas = lbls[3] / 365

            for i in range(batch_size):

                img_id = img_ids[i][0]
                delta = deltas[i][0]

                img_name = str(img_id) + '_{img_type}_'
                img_name += '{:04.1f}.nii.gz'.format(delta)

                img_path = os.path.join(out_dir, img_name)

                def _save_nii(data, img_type):
                    nii = nib.Nifti1Image(data, np.eye(4))
                    path = img_path.format(img_type=img_type)
                    nib.save(nii, path)
                    csv.loc[csv.img_id == img_id, 'img_path_'+img_type] = path
                
                _save_nii(yf[i], 'yf')
                _save_nii(flow[i], 'flow')
                _save_nii(flow_ti[i], 'flow_ti')

                csv.loc[csv.img_id == img_id, 'Df'] = Df[i]
        
        
        csv.to_csv(csv_out_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
       
    parser.add_argument("--img_path", type=str, dest="img_path", default=None)
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=1)
    parser.add_argument("--start", type=float, dest="start", default=None)
    parser.add_argument("--stop", type=float, dest="stop", default=None)
    parser.add_argument("--step", type=float, dest="step", default=None)
    parser.add_argument("--out_dir", type=str, dest="out_dir", default=None)
    parser.add_argument("--gen_model", type=str,
                        dest="gen_model_file", help="path to generator h5 model file")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    
    args = parser.parse_args()
    predict(**vars(args))
