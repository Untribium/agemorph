# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
from datetime import datetime

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.metrics import sparse_categorical_accuracy
from keras.losses import mean_absolute_error, sparse_categorical_crossentropy
from keras.utils import multi_gpu_model, Progbar

# project imports
from src import datagenerators, networks
from src.callbacks import PredictionCallback

def train(csv_path,
          tag,
          gpu_id,
          epochs,
          steps_per_epoch,
          batch_size,
          lr,
          beta_1,
          beta_2,
          epsilon,
          layers,
          batchnorm,
          valid_steps,
          valid_freq):

    """
    model training function
    :param csv_path
    :param model_dir: model folder to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param epochs: number of training iterations
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param bidir: logical whether to use bidirectional cost function
    """

    model_config = locals()

    assert len(layers) % 2 == 0, "layers must have even length"

    assert os.path.isfile(csv_path), 'csv not found at {}'.format(csv_path)
    
    csv_path = os.path.abspath(csv_path)
    model_config['csv_path'] = csv_path

    vol_shape = (80, 96, 80)
    model_config['vol_shape'] = vol_shape

    print('input vol_shape is {}'.format(vol_shape))

    model_dir = './runs/'
    model_dir += 'clf_{:%Y%m%d_%H%M}'.format(datetime.now())
    model_dir += '_gpu={}'.format(str(gpu_id))
    model_dir += '_bs={}'.format(batch_size)
    model_dir += '_lr={}'.format(lr)
    model_dir += '_b1={}'.format(beta_1)
    model_dir += '_b2={}'.format(beta_2)
    model_dir += '_ep={}'.format(epsilon)
    model_dir += '_bn={}'.format(batchnorm)
    model_dir += '_ls={}'.format(layers)
    model_dir += '_tag={}'.format(tag) if tag != '' else ''

    model_dir = model_dir.replace(' ', '')
    model_dir = model_dir.replace(',', '_')

    print('model_dir is {}'.format(model_dir))

    # parse layers
    layers = list(zip(layers[0::2], layers[1::2]))
    print(layers)


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

    # prepare callbacks
    save_file_name = os.path.join(model_dir, 'clf_{epoch:04d}.h5')

    # prepare the model
    with tf.device(gpu):

        # optimizers
        model_opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        # model
        model = networks.classifier_net(vol_shape, layers, batchnorm)

        # save first iteration
        model.save(save_file_name.format(epoch=0))

        model_losses = [sparse_categorical_crossentropy]

    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)


    img_keys = ['img_path']
    lbl_keys = ['pat_dx']

    train_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                              lbl_keys=lbl_keys, batch_size=batch_size,
                              sample=True, split='train')
    
    valid_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                              lbl_keys=lbl_keys, batch_size=batch_size,
                              sample=True, split='eval')
    
    predi_csv_gen = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                              lbl_keys=lbl_keys, batch_size=batch_size,
                              sample=True, split='eval')
   
    train_data = datagenerators.clf_gen(train_csv_gen)
    valid_data = datagenerators.clf_gen(valid_csv_gen)
    predi_data = datagenerators.clf_gen(predi_csv_gen)

    tboard_callback = TensorBoard(log_dir=model_dir)
    tboard_callback.set_model(model)

    pred_callback = PredictionCallback(predi_data, update_freq=10)

    # fit model
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)

        # single gpu
        else:
            save_callback = ModelCheckpoint(save_file_name, period=5)
            mg_model = model

        mg_model.compile(optimizer=model_opt,
                         loss=model_losses,
                         loss_weights=[1],
                         metrics=[sparse_categorical_accuracy])

        mg_model.fit_generator(train_data,
                         epochs=epochs,
                         callbacks=[save_callback, tboard_callback, pred_callback],
                         steps_per_epoch=steps_per_epoch,
                         verbose=1,
                         validation_data=valid_data,
                         validation_steps=valid_steps)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--csv", type=str,
                        dest="csv_path", help="data folder")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--tag", type=str, default='',
                        dest="tag", help="tag added to run dir")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.0001, help="learning rate")
    parser.add_argument("--beta1", type=float,
                        dest="beta_1", default=0.9, help="beta1")
    parser.add_argument("--beta2", type=float,
                        dest="beta_2", default=0.999, help="beta2")
    parser.add_argument("--epsilon", type=float,
                        dest="epsilon", default=0.1, help="epsilon")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=8,
                        help="batch_size")
    parser.add_argument("--layers", dest="layers", type=int, nargs="+",
                        help="pairs of channel, strides for each layer",
                        default=[4,1,16,2,16,1,64,2,64,1,128,2,128,1,256,2,256,1])
    parser.add_argument("--valid_steps", type=int,
                        dest="valid_steps", default=100,
                        help="valid_steps")
    parser.add_argument("--valid_freq", type=int,
                        dest="valid_freq", default=50,
                        help="valid_freq")
    parser.add_argument("--batchnorm", dest="batchnorm", action="store_true")
    parser.set_defaults(batchnorm=False)

    args = parser.parse_args()
    train(**vars(args))
