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

sys.path.append('../../ext/neuron')

import neuron.layers as nrn_layers


def predict(gpu_id, csv_path, split_col, split, batch_size, out_dir, gen_model_file,
                                                 use_cpu, delta, out_imgs):

    # GPU handling
    if gpu_id is not None and not use_cpu:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        print('using cpu')
        gpu = '/cpu:0'

    # check generator model file exists 
    assert os.path.isfile(gen_model_file), "generator model file does not exist"

    gen_model_file = os.path.abspath(gen_model_file)

    # extract run directory and model checkpoint name
    model_dir, model_name = os.path.split(gen_model_file)
    model_name = os.path.splitext(model_name)[0]
    model_name += '_{:02.0f}'.format(delta) if delta is not None else ''

    # load model config
    config_path = os.path.join(model_dir, 'config.pkl')

    assert os.path.isfile(config_path), 'model_config not found'

    if isinstance(split, str):
        split = [split]

    # create out_dir
    if out_dir is None or out_dir == '':
        out_dir = os.path.join(model_dir, 'predict', split_col+'_'+''.join(split), model_name)
    
    print('out_dir:', out_dir)
 
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model_config = pickle.load(open(config_path, 'rb'))

    # extract variables
    vol_shape = model_config['vol_shape']
    max_delta = model_config['max_delta']
    vel_resize = model_config['vel_resize']
    prior_lambda = model_config['prior_lambda']
    image_sigma = model_config['image_sigma']
    loss_weights = model_config['loss_weights']
    enc_nf = model_config['enc_nf']
    dec_nf = model_config['dec_nf']
    int_steps = model_config['int_steps']

    flow_shape = tuple(int(d * vel_resize) for d in vol_shape)

    # use csv used in training if no path is provided
    if csv_path is None or csv_path == '':
        csv_path = model_config['csv_path']

    csv_out_path = os.path.join(out_dir, 'meta.csv')

    if not os.path.isfile(csv_out_path):
        csv = pd.read_csv(csv_path)
        csv = csv[csv[split_col].isin(split)]
        
        # backup delta
        if delta is not None:
            csv['delta_t_real'] = csv['delta_t']
            csv['delta_t'] = delta * 365
        
        # write meta to out_dir
        csv.to_csv(csv_out_path, index=False)

    csv = pd.read_csv(csv_out_path)
 
    img_keys = ['img_path_0', 'img_path_1']
    lbl_keys = ['delta_t', 'pat_dx_1', 'img_id_0', 'img_id_1']
  
    test_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                                    lbl_keys=lbl_keys, batch_size=batch_size,
                                    split=(split_col, split), n_epochs=1,
                                    sample=False, shuffle=False)

    test_data = datagenerators.vae_generator(test_csv_gen, flow_shape, max_delta, int_steps, batch_size)

    with tf.device(gpu):

        print('loading model')

        # get loss class
        loss_class = losses.Miccai2018(image_sigma=image_sigma,
                                       prior_lambda=prior_lambda,
                                       flow_shape=flow_shape)

        # create generator model
        gen_net = networks.miccai2018_net(vol_shape, enc_nf, dec_nf)

        # load weights into model
        gen_net.load_weights(gen_model_file)

        print('starting predict')

        # predict
        for i, (inputs, _, batch) in enumerate(test_data):

            if i % 10 == 0:
                print('step', i)

            # generate
            yf, flow = gen_net.predict(inputs)

            xr_ids = batch['img_id_0']
            yr_ids = batch['img_id_1']

            for i in range(batch_size):

                xr_id = xr_ids[i][0]
                yr_id = yr_ids[i][0]

                index = (csv.img_id_0 == xr_id) & (csv.img_id_1 == yr_id)

                img_name = str(xr_id) + '_' + str(yr_id) + '_{img_type}.nii.gz'
                img_path = os.path.join(out_dir, img_name)

                def _save_nii(data, img_type):
                    nii = nib.Nifti1Image(data, np.eye(4))
                    path = img_path.format(img_type=img_type)
                    nib.save(nii, path)
                    csv.loc[index, 'img_path_'+img_type] = path
               
                if 'yf' in out_imgs: 
                    _save_nii(yf[i], 'yf')

                if 'flow' in out_imgs:
                    _save_nii(flow[i], 'flow')
        
        
        csv.to_csv(csv_out_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
       
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--csv", type=str, dest="csv_path")
    parser.add_argument("--split_col", type=str,
                        dest="split_col", default="split")
    parser.add_argument("--split", type=str, nargs="+",
                        dest="split", default=["test"])
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=1)
    parser.add_argument("--delta", type=float, dest="delta", default=None)
    parser.add_argument("--out_dir", type=str, dest="out_dir", default=None)
    parser.add_argument("--gen_model", type=str,
                        dest="gen_model_file", help="path to generator h5 model file")
    parser.add_argument("--output", dest="out_imgs", nargs="+",
                        default=['yf', 'flow'])
     
    parser.add_argument("--use_cpu", dest="use_cpu", action="store_true")
    
    parser.set_defaults(use_cpu=False)
    
    args = parser.parse_args()
    predict(**vars(args))
