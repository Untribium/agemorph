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


def predict(gpu_id, csv_path, split, batch_size, out_dir, gen_model_file):

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

    # create out_folder
    if out_dir is None or out_dir == '':
        out_dir = os.path.join(model_dir, 'test', model_name)
    
    print('out_dir:', out_dir)
 
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # load model config
    config_path = os.path.join(model_dir, 'config.pkl')
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

    csv = pd.read_csv(csv_path)
    csv = csv[csv.split == split]
 
    img_keys = ['img_path_0', 'img_path_1']
    lbl_keys = ['delta_t', 'img_id_0']
  
    test_csv_data = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                                    lbl_keys=lbl_keys, batch_size=batch_size,
                                    split=split, n_epochs=1, sample=False)

    kl_dummy = np.zeros((batch_size, *flow_shape, len(vol_shape)-1))

    test_data = convert_delta(test_csv_data, max_delta, int_steps, kl_dummy)

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
        for i, (inputs, _) in enumerate(test_data):

            if i % 10 == 0:
                print('step', i)

            # generate
            yf, flow = gen_net.predict(inputs[:2])

            img_ids = inputs[2]

            for i, img_id in enumerate(img_ids):

                img_id = img_id[0]

                img_path = os.path.join(out_dir, str(img_id)+'_{img_type}.nii')

                def _save_nii(data, img_type):
                    nii = nib.Nifti1Image(data, np.eye(4))
                    path = img_path.format(img_type=img_type)
                    nib.save(nii, path)
                    csv.loc[csv.img_id_0 == img_id, 'img_path_'+img_type] = path
                
                _save_nii(yf[i], 'yf')
                _save_nii(flow[i], 'flow')
        
        
        csv_out = os.path.join(out_dir, 'meta.csv')
        csv.to_csv(csv_out, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
       
    # optional arguments
    parser.add_argument("--csv", type=str, dest="csv_path")
    parser.add_argument("--split", type=str, dest="split", default="test")
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=8)
    parser.add_argument("--out_dir", type=str, dest="out_dir", default=None)
    parser.add_argument("--gen_model", type=str,
                        dest="gen_model_file", help="path to generator h5 model file")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    
    args = parser.parse_args()
    predict(**vars(args))
