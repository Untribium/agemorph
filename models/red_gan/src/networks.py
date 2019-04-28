"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the
presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture,
and we encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""

# main imports
import sys

# third party
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, add
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization
from keras.initializers import RandomNormal
import keras.initializers
from functools import partial

# import neuron layers, which will be useful for transforming
sys.path.append('../../ext/neuron')
sys.path.append('../../ext/pynd-lib')
sys.path.append('../../ext/pytools-lib')

import neuron.layers as nrn_layers
from neuron.layers import VecInt, SpatialTransformer
import neuron.utils as nrn_utils


def gan_models(vol_shape, batch_size, loss_class, cri_loss_weights, cri_optimizer, 
               cri_base_nf, gen_loss_weights, gen_optimizer, vel_resize, int_steps,
               reg_model_file):

    vol_shape = tuple(vol_shape)
    
    gen_net = generator_net(vol_shape, vel_resize, int_steps)
    cri_net = critic_net(vol_shape, cri_base_nf)

    """
    nomenclature
    x = img 1st visit
    y = img 2nd visit
    r = real
    f = fake
    a = avg
    D = critic output
    """
   
    # --- regressor (pre-trained) ---
    reg_net = keras.load_model(reg_model_file)
    reg_net.trainable = False

    # --- critic ---

    gen_net.trainable = False # freeze generator in critic

    xr_cri = Input(shape=vol_size + (1,)) # real 1st visit
    yr_cri = Input(shape=vol_size + (1,)) # real 2nd visit
    br_cri = Input(shape=(16,))  # real delta (binary)

    cri_inputs = [xr_cri, yr_cri, br_cri]

    # generate image
    yf_cri, _, _ = gen_net([xr_cri, br_cri])

    # interpolated sample
    ya_cri = RandomWeightedAverage(batch_size=batch_size)([yr_cri, yf_cri])

    Dr_cri = cri_net([xr_cri, yr_cri])
    Df_cri = cri_net([xr_cri, yf_cri])
    Da_cri = cri_net([xr_cri, ya_cri])

    cri_outputs = [Dr_cri, Df_cri, Da_cri]

    cri_model = Model(inputs=cri_inputs, outputs=cri_outputs)

    # keras loss functions have fixed params, so use partial to apply avg
    gradient_loss_partial = partial(loss_class.gradient_penalty_loss, y_avg=ya_cri)
    gradient_loss_partial.__name__ = 'gradient_penalty' # keras requires loss name

    cri_loss = [loss_class.wasserstein_loss,
                loss_class.wasserstein_loss,
                gradient_loss_partial]

    cri_model.compile(loss=cri_loss, optimizer=cri_optimizer, loss_weights=cri_loss_weights)

    # --- generator ---

    cri_net.trainable = False # freeze critic in generator
    gen_net.trainable = True # unfreeze generator

    xr_gen = Input(shape=vol_size + (1,)) # real 1st visit
    br_gen = Input(shape=(16,))  # delta (binary)
    
    gen_inputs = [xr_gen, br_gen]

    # predict y_hat
    yf_gen, flow_gen, f = gen_net([xr_gen, br_gen])
   
    # get wasserstein loss
    Df_gen = cri_net([xr_gen, yf_gen])

    # get age loss
    a_yr_gen = reg_net([yr_gen])
    a_yf_gen = reg_net([yf_gen])
    a_df_gen = subtract([a_yf_gen, a_yr_gen])

    gen_outputs = [Df_gen, a_df_gen, yf_gen, flow_gen, f]
    
    gen_model = Model(inputs=gen_inputs, outputs=gen_outputs)

    gen_loss = [loss_class.wasserstein_loss,
                loss_class.l1_loss,
                loss_class.l1_loss,
                loss_class.kl_loss,
                loss_class.dummy_loss]

    gen_model.compile(loss=gen_loss, optimizer=gen_optimizer, loss_weights=gen_loss_weights)

    return cri_model, gen_model


def unet_core(vol_shape, vel_resize):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_shape: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """

    ndims = len(vol_shape)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)
    
    upsample_layer = getattr(KL, 'UpSampling{}D'.format(ndims))

    print('generator unet:')

    # inputs
    e0 = Input(shape=vol_shape + (1,))
    print(K.int_shape(e0))

    inputs = [e0]

    e1 = conv_block(e0, 8, 1)   # 80 32 80   8
    print(K.int_shape(e1))
    e2 = conv_block(e1, 16, 2)  # 40 16 40  16
    print(K.int_shape(e2))
    e3 = conv_block(e2, 16, 1)  # 40 16 40  16
    print(K.int_shape(e3))
    e4 = conv_block(e3, 32, 2)  # 20  8 20  32
    print(K.int_shape(e4))
    e5 = conv_block(e4, 32, 1)  # 20  8 20  32
    print(K.int_shape(e5))
    e6 = conv_block(e5, 64, 2)  # 10  4 10  64
    print(K.int_shape(e6))
    e7 = conv_block(e6, 64, 1)  # 10  4 10  64
    print(K.int_shape(e7))
    e8 = conv_block(e7, 64, 2)  #  5  2  5  64
    print(K.int_shape(e8))

    e9 = KL.Flatten()(e8)
    print(K.int_shape(e9))

    f = KL.Dense(400)(e9)
    print(K.int_shape(f))

    d9 = KL.Dense(3200)(f)
    print(K.int_shape(d9))

    d8 = #TODO reshape          #  5  2  5  64
    d8 = concatenate([e8, d8])  #  5  2  5 128
    d8 = conv_block(d8, 64, 1)  #  5  2  5  64
    print(K.int_shape(d8))

    d7 = upsample_layer()(d8)   # 10  4 10  64
    print(K.int_shape(d7))
    d7 = concatenate([e7, d7])  # 10  4 10 128
    d6 = conv_block(d7, 32, 1)  # 10  4 10  32
    print(K.int_shape(d6))
 
    d5 = upsample_layer()(d6)   # 20  8 20  32
    print(K.int_shape(d5))
    d5 = concatenate([e5, d5])  # 20  8 20  64
    d4 = conv_block(d5, 16, 1)  # 20  8 20  16
    print(K.int_shape(d4))

    d3 = upsample_layer()(d4)   # 40 16 40  16
    print(K.int_shape(d3))
    d3 = concatenate([e3, d3])  # 40 16 40  32
    d2 = conv_block(d3, 8, 1)   # 40 16 40   8
    print(K.int_shape(d2))

    if vel_resize == 1.0:
        d1 = upsample_layer()(d2)
        print(K.int_shape(d1))
        d1 = concatenate([e1, d1])
        d0 = conv_block(d1, 3, 1)
        print(K.int_shape(d0))
        
        outputs = [d0, f]
    else:
        outputs = [d2, f]
    
    return Model(inputs=inputs, outputs=outputs)


def generator_net(vol_shape, vel_resize, int_steps):
 
    ndims = len(vol_shape)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)
    
    unet = unet_core(vol_shape, vel_resize)
    
    # target delta in binary representation
    b_in = Input(shape=(16,)) 
    
    x_in = unet.inputs[0]
    x_out, f = unet.outputs

    inputs = [x_in, b_in] 
    
    Conv = getattr(KL, 'Conv{}D'.format(ndims))

    flow_mean = Conv(ndims, kernel_size=3, padding='same', name='flow_mean',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x_out)

    flow_log_sigma = Conv(ndims, kernel_size=3, padding='same', name='flow_log_sigma',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                bias_initializer=keras.initializers.Constant(value=-10))(x_out)

    flow_params = concatenate([flow_mean, flow_log_sigma])

    # sample velocity field (using reparameterization)
    z_sample = Sample(name='z_sample')([flow_mean, flow_log_sigma])

    # integrate flow
    flow = VecInt(method='ss', name='flow_int', int_steps=int_steps)([z_sample, b_in])

    # resize to full size 
    if vel_resize != 1.0:
        flow = trf_resize(flow, vel_resize, name='flow_resize')

    y = SpatialTransformer(interp_method='linear', indexing='ij')([x_in, flow])
  
    outputs = [y, flow_params, f]

    return Model(inputs=inputs, outputs=outputs)


def critic_net(vol_shape, base_nf=8):

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    x = Input(shape=vol_shape + (1,)) # image 1st visit
    y = Input(shape=vol_shape + (1,)) # image 2nd visit (real or fake)

    inputs = [x, y]

    print('critic net:')

    x = concatenate([x, y])
    print(K.int_shape(x))

    x = conv_block(x, base_nf, 2)
    print(K.int_shape(x))

    x = conv_block(x, base_nf*2, 2)
    print(K.int_shape(x))
    x = conv_block(x, base_nf*2)
    print(K.int_shape(x))
   
    x = conv_block(x, base_nf*4, 2)
    print(K.int_shape(x))
    x = conv_block(x, base_nf*4)
    print(K.int_shape(x))
    
    x = conv_block(x, base_nf*8, 2)
    print(K.int_shape(x))
    x = conv_block(x, base_nf*8)
    print(K.int_shape(x))
    
    x = conv_block(x, base_nf*16)
    print(K.int_shape(x))
    x = conv_block(x, base_nf*16)
    print(K.int_shape(x))
    
    x = conv_block(x, 1, kernel_size=1, activation=False)
    print(K.int_shape(x))

    x = KL.Flatten()(x)
    print(K.int_shape(x))

    # weighted sum
    x = KL.Dense(1, use_bias=False)(x)
    print(K.int_shape(x))
    
    outputs = [x]
 
    return Model(inputs=inputs, outputs=outputs)


# Helper functions
def conv_block(x_in, nf, strides=1, kernel_size=3, activation=True, batchnorm=False):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    Conv = getattr(KL, 'Conv{}D'.format(ndims))
    
    x_out = Conv(nf, kernel_size=kernel_size, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    
    if activation:
        x_out = LeakyReLU(0.2)(x_out)

    if batchnorm:
        x_out = BatchNormalization()(x_out)
    
    return x_out


def maxpool_block(x_in, strides=2, pool_size=2):
    """
    maxpool module
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    MaxPooling = getattr(KL, 'MaxPooling{}D'.format(ndims))

    strides = (strides,) * ndims
    pool_size = (pool_size,) * ndims

    x_out = MaxPooling(pool_size=pool_size, strides=strides, padding='same')(x_in)

    return x_out


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


def trf_resize(trf, vel_resize, name='flow'):
    if vel_resize > 1:
        trf = nrn_layers.Resize(1/vel_resize, name=name+'_tmp')(trf)
        return Rescale(1 / vel_resize, name=name)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = Rescale(1 / vel_resize, name=name+'_tmp')(trf)
        return nrn_layers.Resize(1/vel_resize, name=name)(trf)


class Sample(Layer):
    """ 
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class RandomWeightedAverage(Layer):

    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(RandomWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)

    def call(self, x):
        ndims = len(K.int_shape(x[0]))-2

        weights = K.random_uniform((self.batch_size,) + (1,) * (ndims+1))

        return weights * x[0] + (1 - weights) * x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Negate(Layer):
    """ 
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


class Rescale(Layer):
    """ 
    Keras layer: rescale data by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize 

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleDouble(Rescale):
    def __init__(self, **kwargs):
        self.resize = 2
        super(RescaleDouble, self).__init__(self.resize, **kwargs)


class ResizeDouble(nrn_layers.Resize):
    def __init__(self, **kwargs):
        self.zoom_factor = 2
        super(ResizeDouble, self).__init__(self.zoom_factor, **kwargs)