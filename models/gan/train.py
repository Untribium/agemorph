# python imports
import os
import sys
import random
from datetime import datetime
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import pandas as pd
import numpy as np
import keras as K
import pickle
from keras.backend.tensorflow_backend import set_session, get_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model, Progbar

# project imports
from src import datagenerators, networks, losses
from src.callbacks import TensorBoardExt, TensorBoardVal
from src.utils import normalize, convert_delta


def train(csv_path,
          tag,
          gpu_id,
          epochs,
          steps_per_epoch,
          batch_size,
          int_steps,
          vel_resize,
          ti_flow,
          lr,
          beta_1,
          beta_2,
          epsilon,
          prior_lambda,
          enc_nf,
          dec_nf,
          cri_base_nf,
          gen_loss_weights,
          cri_loss_weights,
          cri_steps,
          cri_retune_freq,
          cri_retune_steps,
          valid_freq,
          valid_steps):
    
    """
    model training function
    :param csv_path: path to data csv (img paths, labels)
    :param tag: tag for the run, added to run_dir
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param epochs: number of training iterations
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param prior_lambda: the prior_lambda, the scalar in front of the smoothing laplacian, in MICCAI paper
    """

    model_config = locals()

    # gpu handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    vol_shape = (80, 96, 80)

    print('input vol_shape is {}'.format(vol_shape))
    
    assert os.path.isfile(csv_path), 'csv not found at {}'.format(csv_path)

    csv_path = os.path.abspath(csv_path)

    model_config['csv_path'] = csv_path

    model_dir = 'runs/'
    model_dir += 'gan_{:%Y%m%d_%H%M}'.format(datetime.now())
    model_dir += '_gpu={}'.format(str(gpu_id))
    model_dir += '_bs={}'.format(batch_size)
    model_dir += '_enc={}'.format(enc_nf)
    model_dir += '_dec={}'.format(dec_nf)
    model_dir += '_cbn={}'.format(cri_base_nf)
    model_dir += '_lr={}'.format(lr)
    model_dir += '_b1={}'.format(beta_1)
    model_dir += '_b2={}'.format(beta_2)
    model_dir += '_ep={}'.format(epsilon)
    model_dir += '_pl={}'.format(prior_lambda)
    model_dir += '_vr={}'.format(vel_resize)
    model_dir += '_ti={}'.format(ti_flow)
    model_dir += '_is={}'.format(int_steps)
    model_dir += '_cs={}'.format(cri_steps)
    model_dir += '_rf={}'.format(cri_retune_freq)
    model_dir += '_rs={}'.format(cri_retune_steps)
    model_dir += '_glw={}'.format(gen_loss_weights)
    model_dir += '_clw={}'.format(cri_loss_weights)
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

    # prepare the model
    with tf.device(gpu):
        
        # load models
        loss_class = losses.GANLosses(prior_lambda=prior_lambda, flow_shape=flow_shape)

        cri_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        gen_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        cri_model, gen_model = networks.gan_models(
                                        vol_shape, batch_size, loss_class,
                                        cri_loss_weights=cri_loss_weights,
                                        cri_optimizer=cri_optimizer,
                                        gen_loss_weights=gen_loss_weights,
                                        gen_optimizer=gen_optimizer,
                                        enc_nf=enc_nf, dec_nf=dec_nf,
                                        cri_base_nf=cri_base_nf,
                                        vel_resize=vel_resize,
                                        ti_flow=ti_flow,
                                        int_steps=int_steps)
      
        cri_model_save_path = os.path.join(model_dir, 'cri_{:03d}.h5')
        gen_model_save_path = os.path.join(model_dir, 'gen_{:03d}.h5')
 
        # save inital models
        cri_model.save(cri_model_save_path.format(0))
        gen_model.save(gen_model_save_path.format(0))
       

    # data generator
    num_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, num_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, num_gpus)

    # load csv
    csv = pd.read_csv(csv_path)
    
    # get max_delta from csv and store in config
    # max_delta and int_steps determine the resolution of the flow integration
    # e.g. max_delta=6, int_steps=5 results in a resolution of about 5 weeks
    # max_steps = 2**(int_steps+1)-1 = 63, 6 years = 72 months
    max_delta = csv['delta_t'].max()
    model_config['max_delta'] = max_delta
    
    # csv columns for img paths and labels
    img_keys = ['img_path_0', 'img_path_1']
    lbl_keys = ['delta_t']

    # datagens for training and validation
    train_csv_data = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, split='train')

    valid_csv_data = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, split='eval')

    # convert the delta to channel (for critic) and bin_repr (for ss in gen)
    train_data = convert_delta(train_csv_data, max_delta, int_steps)
    valid_data = convert_delta(valid_csv_data, max_delta, int_steps)

    # write model_config to run_dir
    config_path = os.path.join(model_dir, 'config.pkl')
    pickle.dump(model_config, open(config_path, 'wb'))


    # labels for train/predict
    # dummy tensor for kl loss, must have correct flow shape
    kl_dummy = np.zeros((batch_size, *flow_shape, len(vol_shape)-1))
   
    # labels for critic ws loss
    real = np.ones((batch_size, 1)) * (-1) # real labels
    fake = np.ones((batch_size, 1))        # fake labels
    avgd = np.ones((batch_size, 1))        # dummy labels for gradient penalty

 
    # tboard callbacks
    tboard_train = TensorBoardExt(log_dir=model_dir)
    tboard_train.set_model(gen_model)

    tboard_valid = TensorBoardVal(log_dir=valid_dir, data=valid_data,
                                  cri_model=cri_model, gen_model=gen_model,
                                  freq=valid_freq, steps=valid_steps,
                                  batch_size=batch_size, kl_dummy=kl_dummy)
    tboard_valid.set_model(gen_model)


    # fit generator
    with tf.device(gpu):

        abs_step = 0

        for epoch in range(epochs):
            
            print('epoch {}/{}'.format(epoch, epochs))

            cri_steps_ep = cri_steps

            # check if retune epoch, if so adjust critic steps
            if epoch % cri_retune_freq == 0:
                cri_steps_ep = cri_retune_steps
                print('retuning critic')

            progress_bar = Progbar(target=steps_per_epoch)

            for step in range(steps_per_epoch):
                
                # train critic
                for c_step in range(cri_steps_ep):
                    
                    imgs, lbls = next(train_data)

                    cri_in = [imgs[0], imgs[1], lbls[0], lbls[1]] # xr, yr, dr, db
                    cri_true = [real, fake, avgd]

                    cri_logs = cri_model.train_on_batch(cri_in, cri_true)

                imgs, lbls = next(train_data)

                gen_in = [imgs[0], lbls[0], lbls[1]] # xr, dr, db
                gen_true = [imgs[0], kl_dummy, kl_dummy, real]

                # train generator
                gen_logs = gen_model.train_on_batch(gen_in, gen_true)

                # update tensorboard
                tboard_train.on_epoch_end(abs_step, cri_logs, gen_logs)
                tboard_valid.on_epoch_end(abs_step)
                #tensorboard_summaries(tboard_train, abs_step, cri_logs, gen_logs)

                abs_step += 1
                progress_bar.add(1)

            if epoch % 5 == 0:
                cri_model.save(cri_model_save_path.format(epoch))
                gen_model.save(gen_model_save_path.format(epoch))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--csv", type=str, dest='csv_path',
                        help="path to data csv")
    parser.add_argument("--tag", type=str,
                        dest="tag", default='',
                        help="tag to be added to model_dir")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=4,
                        help="batch_size")
    parser.add_argument("--vel_resize", type=float,
                        dest="vel_resize", default=1.0,
                        help="vel_resize")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--beta1", type=float,
                        dest="beta_1", default=0.0, help="optimizer beta1")
    parser.add_argument("--beta2", type=float,
                        dest="beta_2", default=0.9, help="optimizer beta2")
    parser.add_argument("--epsilon", type=float,
                        dest="epsilon", default=0.1, help="epsilon")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=1000,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=25,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--enc_nf", type=int, nargs="+",
                        dest="enc_nf", default=[16, 32, 32, 32])
    parser.add_argument("--dec_nf", type=int, nargs="+",
                        dest="dec_nf", default=[32, 32, 32, 32, 16, 8])
    parser.add_argument("--cri_base_nf", type=int,
                        dest="cri_base_nf", default=16)
    parser.add_argument("--gen_loss_weights", type=float, nargs="+",
                        dest="gen_loss_weights", default=[1, 100, 10, 1])
    parser.add_argument("--cri_loss_weights", type=float, nargs="+",
                        dest="cri_loss_weights", default=[1, 1, 10])
    parser.add_argument("--int_steps", type=int,
                        dest="int_steps", default=5,
                        help="number of integration steps in scaling and squaring")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="number of steps per epoch")
    parser.add_argument("--cri_steps", type=int,
                        dest="cri_steps", default=5,
                        help="number of critic steps per generator step")
    parser.add_argument("--cri_retune_freq", type=int,
                        dest="cri_retune_freq", default=10,
                        help="frequency of critic retune epochs")
    parser.add_argument("--cri_retune_steps", type=int,
                        dest="cri_retune_steps", default=25,
                        help="number of critic steps in retune epochs")
    parser.add_argument("--valid_freq", type=int,
                        dest="valid_freq", default=25,
                        help="frequency of validation")
    parser.add_argument("--valid_steps", type=int,
                        dest="valid_steps", default=5,
                        help="number of validation steps")
    parser.add_argument("--ti_flow", dest="ti_flow", action="store_true")
    parser.set_defaults(ti_flow=False)
    
    args = parser.parse_args()
    train(**vars(args))
