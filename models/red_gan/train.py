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
from src.utils import normalize


def train(csv_path,
          tag,
          gpu_id,
          epochs,
          steps_per_epoch,
          batch_size,
          vol_shape,
          int_steps,
          vel_resize,
          sample_weights,
          lr,
          beta_1,
          beta_2,
          epsilon,
          prior_lambda,
          batchnorm,
          leaky,
          split_col,
          split_train,
          split_eval,
          reg_model_file,
          clf_model_file,
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

    # grab config (all local variables at this point)
    model_config = locals()

    # claim gpu, do early so we fail early if it's occupied
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # convert vol_shape (is list)
    vol_shape = tuple(vol_shape)
    model_config['vol_shape'] = vol_shape 
    
    print('input vol_shape is {}'.format(vol_shape))

    # check csv exists
    assert os.path.isfile(csv_path), 'csv not found at {}'.format(csv_path)

    # add csv path to config 
    csv_path = os.path.abspath(csv_path)
    model_config['csv_path'] = csv_path

    # check regressor model file exists 
    if reg_model_file:
        msg = 'reg model file not found at {}'.format(reg_model_file)
        assert os.path.isfile(reg_model_file), msg
        reg_model_file = os.path.abspath(reg_model_file) 
        model_config['reg_model_file'] = reg_model_file

    # check classifier model file exists 
    if clf_model_file:
        msg = 'clf model file not found at {}'.format(clf_model_file)
        assert os.path.isfile(clf_model_file), msg
        clf_model_file = os.path.abspath(clf_model_file)
        model_config['clf_model_file'] = clf_model_file

    # stitch together run_dir name
    run_dir = 'runs/'
    run_dir += 'gan_{:%Y%m%d_%H%M}'.format(datetime.now())
    run_dir += '_gpu={}'.format(str(gpu_id))
    run_dir += '_bs={}'.format(batch_size)
    run_dir += '_cl={}'.format(cri_base_nf)
    run_dir += '_lr={}'.format(lr)
    run_dir += '_b1={}'.format(beta_1)
    run_dir += '_b2={}'.format(beta_2)
    run_dir += '_ep={}'.format(epsilon)
    run_dir += '_pl={}'.format(prior_lambda)
    run_dir += '_lk={}'.format(leaky)
    run_dir += '_bn={}'.format(batchnorm)
    run_dir += '_vr={}'.format(vel_resize)
    run_dir += '_is={}'.format(int_steps)
    run_dir += '_cs={}'.format(cri_steps)
    run_dir += '_rf={}'.format(cri_retune_freq)
    run_dir += '_rs={}'.format(cri_retune_steps)
    run_dir += '_sw={}'.format(sample_weights is not None)
    run_dir += '_reg={}'.format(reg_model_file is not None)
    run_dir += '_clf={}'.format(clf_model_file is not None)
    run_dir += '_glw={}'.format(gen_loss_weights)
    run_dir += '_clw={}'.format(cri_loss_weights)
    run_dir += '_tag={}'.format(tag) if tag != '' else ''
    
    run_dir = run_dir.replace(' ', '')
    run_dir = run_dir.replace(',', '_')

    print('run_dir is {}'.format(run_dir))

    # calculate flow_shape given the resize param
    flow_shape = tuple(int(d * vel_resize) for d in vol_shape)

    # create run dirs
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)

    valid_dir = os.path.join(run_dir, 'eval')

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
                                        cri_base_nf=cri_base_nf,
                                        gen_loss_weights=gen_loss_weights,
                                        gen_optimizer=gen_optimizer,
                                        vel_resize=vel_resize,
                                        int_steps=int_steps,
                                        reg_model_file=reg_model_file,
                                        clf_model_file=clf_model_file,
                                        batchnorm=batchnorm,
                                        leaky=leaky)
      
        cri_model_save_path = os.path.join(run_dir, 'cri_{:03d}.h5')
        gen_model_save_path = os.path.join(run_dir, 'gen_{:03d}.h5')
 
        # save inital models
        cri_model.save(cri_model_save_path.format(0))
        gen_model.save(gen_model_save_path.format(0))

    # load csv
    csv = pd.read_csv(csv_path)
    
    # get max_delta from csv and store in config
    # max_delta and int_steps determine the resolution of the flow integration
    # e.g. max_delta=6y, int_steps=5 results in a resolution of about 5 weeks
    # max_steps = 2**(int_steps+1)-1 = 63, 6 years = 72 months
    max_delta = csv['delta_t'].max()
    model_config['max_delta'] = max_delta
    
    # csv columns for img paths and labels
    img_keys = ['img_path_0', 'img_path_1']
    lbl_keys = ['delta_t', 'pat_dx_1']

    # datagens for training and validation
    train_csv_data = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, weights=sample_weights,
                            split=(split_col, split_train))

    valid_csv_data = datagenerators.csv_gen(csv_path, img_keys=img_keys,
                            lbl_keys=lbl_keys, batch_size=batch_size,
                            sample=True, weights=sample_weights,
                            split=(split_col, split_eval))

    use_reg = reg_model_file is not None
    use_clf = clf_model_file is not None

    cri_train_data, gen_train_data = datagenerators.gan_generators(
                                    csv_gen=train_csv_data, vol_shape=vol_shape,
                                    flow_shape=flow_shape, max_delta=max_delta,
                                    int_steps=int_steps, use_reg=use_reg,
                                    use_clf=use_clf)

    cri_valid_data, gen_valid_data = datagenerators.gan_generators(
                                    csv_gen=valid_csv_data, vol_shape=vol_shape,
                                    flow_shape=flow_shape, max_delta=max_delta,
                                    int_steps=int_steps, use_reg=use_reg,
                                    use_clf=use_clf)

    # write model_config to run_dir
    config_path = os.path.join(run_dir, 'config.pkl')
    pickle.dump(model_config, open(config_path, 'wb'))
    
    print('model_config:')
    print(model_config)

    # tboard callbacks
    tboard_train = TensorBoardExt(log_dir=run_dir, use_reg=use_reg, use_clf=use_clf)
    tboard_train.set_model(gen_model)

    tboard_valid = TensorBoardVal(log_dir=valid_dir, use_reg=use_reg, use_clf=use_clf,
                              cri_data=cri_valid_data, gen_data=gen_valid_data,
                              cri_model=cri_model, gen_model=gen_model,
                              freq=valid_freq, steps=valid_steps)
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
                   
                    inputs, labels, _ = next(cri_train_data)
 
                    cri_logs = cri_model.train_on_batch(inputs, labels)

                inputs, labels, _ = next(gen_train_data)

                # train generator
                gen_logs = gen_model.train_on_batch(inputs, labels)

                # update tensorboard
                tboard_train.on_epoch_end(abs_step, cri_logs, gen_logs)
                tboard_valid.on_epoch_end(abs_step)

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
                        help="tag to be added to run_dir")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--sample_weights", type=str, default=None,
                        dest="sample_weights", help="sample weight column in csv")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=4, help="batch_size")
    parser.add_argument("--vol_shape", type=int, nargs="+",
                        dest="vol_shape", default=[80, 32, 80])
    parser.add_argument("--vel_resize", type=float,
                        dest="vel_resize", default=1.0, help="vel_resize")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--beta1", type=float,
                        dest="beta_1", default=0.0, help="optimizer beta1")
    parser.add_argument("--beta2", type=float,
                        dest="beta_2", default=0.9, help="optimizer beta2")
    parser.add_argument("--epsilon", type=float,
                        dest="epsilon", default=0.1, help="epsilon")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=1000, help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=25,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--reg_model", type=str,
                        dest="reg_model_file")
    parser.add_argument("--clf_model", type=str,
                        dest="clf_model_file")
    parser.add_argument("--cri_base_nf", type=int,
                        dest="cri_base_nf", default=8)
    parser.add_argument("--gen_loss_weights", type=float, nargs="+",
                        dest="gen_loss_weights", default=[1, 200, 5, 100, 0, 0])
    parser.add_argument("--cri_loss_weights", type=float, nargs="+",
                        dest="cri_loss_weights", default=[1, 1, 10])
    parser.add_argument("--int_steps", type=int,
                        dest="int_steps", default=6,
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
                        dest="cri_retune_steps", default=5,
                        help="number of critic steps in retune epochs")
    parser.add_argument("--valid_freq", type=int,
                        dest="valid_freq", default=25,
                        help="frequency of validation")
    parser.add_argument("--valid_steps", type=int,
                        dest="valid_steps", default=5,
                        help="number of validation steps")
    parser.add_argument("--leaky", type=float, default=0.2,
                        dest="leaky")
    parser.add_argument("--split_col", type=str,
                        dest="split_col", default='split')
    parser.add_argument("--split_train", type=str, nargs="+",
                        dest="split_train", default=['train'])
    parser.add_argument("--split_eval", type=str, nargs="+",
                        dest="split_eval", default=['eval'])

    parser.add_argument("--batchnorm", dest="batchnorm", action="store_true")
    
    parser.set_defaults(batchnorm=False)
    
    args = parser.parse_args()
    train(**vars(args))
