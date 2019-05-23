# python imports
import os
import glob
import sys
import random
from datetime import datetime
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model 

# project imports
from src import datagenerators, networks, losses
from src.callbacks import TensorBoardExt

sys.path.append('../../ext/neuron')

import neuron.callbacks as nrn_gen

def train(csv_path,
          tag,
          gpu_id,
          epochs,
          steps_per_epoch,
          batch_size,
          int_steps,
          vel_resize,
          lr,
          prior_lambda,
          image_sigma,
          enc_nf,
          dec_nf,
          vol_shape,
          loss_weights,
          split_col,
          split_train,
          split_eval):
    
    """
    model training function
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param epochs: number of training iterations
    :param prior_lambda: the prior_lambda, the scalar in front of the smoothing laplacian, in MICCAI paper
    :param image_sigma: the image sigma in MICCAI paper
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    """

    model_config = locals()
    
    vol_shape = tuple(vol_shape)
    model_config['vol_shape'] = vol_shape 
    print('input vol_shape is {}'.format(vol_shape))
    
    assert os.path.isfile(csv_path), 'csv not found at {}'.format(csv_path)

    model_dir = './runs/'
    model_dir += 'vae_{:%Y%m%d_%H%M}'.format(datetime.now())
    model_dir += '_gpu={}'.format(str(gpu_id))
    model_dir += '_bs={}'.format(batch_size)
    model_dir += '_enc={}'.format(enc_nf)
    model_dir += '_dec={}'.format(dec_nf)
    model_dir += '_lr={}'.format(lr)
    model_dir += '_pl={}'.format(prior_lambda)
    model_dir += '_is={}'.format(image_sigma)
    model_dir += '_vr={}'.format(vel_resize)
    model_dir += '_lw={}'.format(loss_weights)
    model_dir += '_tag={}'.format(tag) if tag != '' else ''
    
    model_dir = model_dir.replace(' ', '')
    model_dir = model_dir.replace(',', '_')

    print('model_dir is {}'.format(model_dir))

    flow_shape = tuple(int(d * vel_resize) for d in vol_shape)

    valid_dir = os.path.join(model_dir, 'eval')

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if not os.path.isdir(valid_dir):
        os.mkdir(valid_dir)

    # gpu handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):

        model = networks.miccai2018_net(vol_shape, enc_nf, dec_nf)

        save_file_name = os.path.join(model_dir, 'gen_{epoch:03d}.h5')

        # save first iteration
        model.save(save_file_name.format(epoch=0))

        # compile
        loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_shape=flow_shape)
        
        model_losses = [loss_class.recon_loss, loss_class.kl_loss]
        
    
    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    csv = pd.read_csv(csv_path)

    img_keys = ['img_path_0', 'img_path_1']
    lbl_keys = ['delta_t', 'pat_dx_1']

    max_delta = csv['delta_t'].max()

    model_config['max_delta'] = max_delta

    train_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, weights='weight',
                            split=(split_col, split_train))

    valid_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, split=(split_col, split_eval))
    
    board_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, weights='weight',
                            split=(split_col, split_eval))

    train_data = datagenerators.vae_generator(train_csv_gen, flow_shape, max_delta, int_steps)
    valid_data = datagenerators.vae_generator(valid_csv_gen, flow_shape, max_delta, int_steps)
    board_data = datagenerators.vae_generator(board_csv_gen, flow_shape, max_delta, int_steps)
   
    # write model_config
    config_path = os.path.join(model_dir, 'config.pkl')
    pickle.dump(model_config, open(config_path, 'wb'))
 
    # prepare callbacks
    tboard_callback = TensorBoardExt(log_dir=model_dir, valid_data=board_data,
                                                        int_steps=int_steps)

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)
        
        # single gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model

        mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)

        tboard_callback.set_model(mg_model)

        callbacks = [save_callback, tboard_callback]
        
        mg_model.fit_generator(train_data, 
                               epochs=epochs,
                               callbacks=callbacks,
                               steps_per_epoch=steps_per_epoch,
                               verbose=1,
                               validation_data=valid_data,
                               validation_steps=25)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--csv", type=str, dest="csv_path")
    parser.add_argument("--tag", type=str, dest="tag", default='')
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--vel_resize", type=float,
                        dest="vel_resize", default=1.0)
    parser.add_argument("--int_steps", type=int,
                        dest="int_steps", default=5)
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=100,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--image_sigma", type=float,
                        dest="image_sigma", default=0.01,
                        help="image noise parameter")
    parser.add_argument("--vol_shape", type=int, nargs="+",
                        dest="vol_shape", default=[80, 32, 80])
    parser.add_argument("--enc_nf", type=int, nargs="+",
                        dest="enc_nf", default=[16, 32, 32, 32])
    parser.add_argument("--dec_nf", type=int, nargs="+",
                        dest="dec_nf", default=[32, 32, 32, 32, 16, 8])
    parser.add_argument("--loss_weights", type=float, nargs="+",
                        dest="loss_weights", default=[1, 1])
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=4,
                        help="batch_size")
    parser.add_argument("--split_col", type=str,
                        dest="split_col", default="split")
    parser.add_argument("--split_train", type=str, nargs="+",
                        dest="split_train", default=['train'])
    parser.add_argument("--split_eval", type=str, nargs="+",
                        dest="split_eval", default=['eval'])
    
    args = parser.parse_args()
    train(**vars(args))
