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


def gan_models(vol_size, batch_size, loss_class, cri_loss_weights,
               cri_optimizer, cri_base_nf, gen_loss_weights, gen_optimizer,
               enc_nf, dec_nf, vel_resize, ti_flow, int_steps):

    vol_size = tuple(vol_size)
    
    gen_net = generator_net(vol_size, enc_nf, dec_nf, vel_resize, ti_flow, int_steps)
    cri_net = critic_net(vol_size, cri_base_nf)

    """
    nomenclature
    x = img 1st visit
    y = img 2nd visit
    r = real
    f = fake
    a = avg
    D = critic output
    """
    
    # --- critic ---

    gen_net.trainable = False # freeze generator in critic

    xr_cri = Input(shape=vol_size + (1,)) # real 1st visit
    yr_cri = Input(shape=vol_size + (1,)) # real 2nd visit
    dr_cri = Input(shape=vol_size + (1,)) # real delta (channel)
    br_cri = Input(shape=(16,))  # real delta (binary)

    cri_inputs = [xr_cri, yr_cri, dr_cri, br_cri]

    # generate image
    yf_cri, _, _ = gen_net([xr_cri, br_cri])

    # interpolated sample
    ya_cri = RandomWeightedAverage(batch_size=batch_size)([yr_cri, yf_cri])

    Dr_cri = cri_net([xr_cri, yr_cri, dr_cri])
    Df_cri = cri_net([xr_cri, yf_cri, dr_cri])
    Da_cri = cri_net([xr_cri, ya_cri, dr_cri])

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
    dr_gen = Input(shape=vol_size + (1,)) # delta (channel)
    br_gen = Input(shape=(16,))  # delta (binary)
    
    gen_inputs = [xr_gen, dr_gen, br_gen]

    yf_gen, flow_gen, flow_ti_gen = gen_net([xr_gen, br_gen])
    
    Df_gen = cri_net([xr_gen, yf_gen, dr_gen])

    gen_outputs = [yf_gen, flow_gen, flow_ti_gen, Df_gen]
    
    gen_model = Model(inputs=gen_inputs, outputs=gen_outputs)

    gen_loss = [loss_class.l1_loss,
                loss_class.kl_loss,
                loss_class.prec_loss,
                loss_class.wasserstein_loss]

    gen_model.compile(loss=gen_loss, optimizer=gen_optimizer, loss_weights=gen_loss_weights)

    return cri_model, gen_model


def unet_core(vol_size, enc_nf, dec_nf, vel_resize, ti_flow):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)
    
    upsample_layer = getattr(KL, 'UpSampling{}D'.format(ndims))

    # inputs
    x_in = Input(shape=vol_size + (1,))

    inputs = [x_in]

    print('generator unet:')

    print(K.int_shape(x_in))

    # downsample path (encoder)
    x_enc = [x_in]

    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))
        print(K.int_shape(x_enc[-1]))

    # upsample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    print(K.int_shape(x))
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    print(K.int_shape(x))
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    print(K.int_shape(x))

    x_ti = conv_block(x, dec_nf[4])
    print(K.int_shape(x_ti), '(ti)')

    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    print(K.int_shape(x))
    x = conv_block(x, dec_nf[4])
    print(K.int_shape(x))
    
    # upsample to full dim if vel_resize == 1.0
    if vel_resize == 1.0:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])
        print(K.int_shape(x))

    outputs = [x, x_ti]

    return Model(inputs=inputs, outputs=outputs)


def generator_net(vol_size, enc_nf, dec_nf, vel_resize, ti_flow, int_steps):
 
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)
    
    unet = unet_core(vol_size, enc_nf, dec_nf, vel_resize, ti_flow)
    
    # target delta in binary representation
    b_in = Input(shape=(16,)) 
    
    x_in = unet.inputs[0]
    x_out, x_ti = unet.outputs

    inputs = [x_in, b_in] 
    
    Conv = getattr(KL, 'Conv{}D'.format(ndims))

    # time dependent flow component (i.e. aging)
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

    # time independent flow component
    flow_ti = Conv(ndims, kernel_size=3, padding='same', name='flow_ti')(x_ti)
    flow_ti = trf_resize(flow_ti, 0.25, name='flow_ti_resize')
    
    if ti_flow:
        flow = add([flow, flow_ti])

    y = SpatialTransformer(interp_method='linear', indexing='ij')([x_in, flow])
  
    outputs = [y, flow_params, flow_ti]

    return Model(inputs=inputs, outputs=outputs)


def critic_net(vol_size, base_nf=16):

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    x = Input(shape=vol_size + (1,)) # image 1st visit
    y = Input(shape=vol_size + (1,)) # image 2nd visit (real/fake)
    d = Input(shape=vol_size + (1,)) # delta

    inputs = [x, y, d]

    print('critic net:')

    x = concatenate([x, y, d])
    print(K.int_shape(x))

    x = conv_block(x, base_nf, 2)
    print(K.int_shape(x))

    x = conv_block(x, base_nf*2, 2)
    print(K.int_shape(x))
    
    x = conv_block(x, base_nf*4)
    print(K.int_shape(x))
    x = conv_block(x, base_nf*4, 2)
    print(K.int_shape(x))
    
    x = conv_block(x, base_nf*8)
    print(K.int_shape(x))
    x = conv_block(x, base_nf*8, 2)
    print(K.int_shape(x))
    
    x = conv_block(x, base_nf*16)
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


"""
voxelmorph nets
"""

def cvpr2018_net(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    # get the core model
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src, tgt] = unet_model.inputs
    x = unet_model.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # warp the source with the flow
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model


def miccai2018_net(vol_size, enc_nf, dec_nf, int_steps=7, use_miccai_int=False, indexing='ij', bidir=False, vel_resize=1/2):
    """
    architecture for probabilistic diffeomoprhic VoxelMorph presented in the MICCAI 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    The stationary velocity field operates in a space (0.5)^3 of vol_size for computational reasons.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6, see unet function.
    :param use_miccai_int: whether to use the manual miccai implementation of scaling and squaring integration
            note that the 'velocity' field outputted in that case was 
            since then we've updated the code to be part of a flexible layer. see neuron.layers.VecInt
            **This param will be phased out (set to False behavior)**
    :param int_steps: the number of integration steps
    :param indexing: xy or ij indexing. we recommend ij indexing if training from scratch. 
            miccai 2018 runs were done with xy indexing.
            **This param will be phased out (set to 'ij' behavior)**
    :return: the keras model
    """    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=False)
    [src, tgt] = unet_model.inputs
    x_out = unet_model.outputs[-1]

    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x_out)
    # we're going to initialize the velocity variance very low, to start stable.
    flow_log_sigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10),
                            name='log_sigma')(x_out)
    flow_params = concatenate([flow_mean, flow_log_sigma])

    # velocity sample
    flow = Sample(name="z_sample")([flow_mean, flow_log_sigma])

    # integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
    if use_miccai_int:
        # for the miccai2018 submission, the squaring layer
        # scaling was essentially built in by the network
        # was manually composed of a Transform and and Add Layer.
        v = flow
        for _ in range(int_steps):
            v1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([v, v])
            v = keras.layers.add([v, v1])
        flow = v

    else:
        # new implementation in neuron is cleaner.
        z_sample = flow
        flow = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(z_sample)
        if bidir:
            rev_z_sample = Negate()(z_sample)
            neg_flow = nrn_layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(rev_z_sample)

    # get up to final resolution
    flow = trf_resize(flow, vel_resize, name='diffflow')

    if bidir:
        neg_flow = trf_resize(neg_flow, vel_resize, name='neg_diffflow')

    # transform
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    if bidir:
        y_tgt = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([tgt, neg_flow])

    # prepare outputs and losses
    outputs = [y, flow_params]
    if bidir:
        outputs = [y, y_tgt, flow_params]

    # build the model
    return Model(inputs=[src, tgt], outputs=outputs)


def nn_trf(vol_size, indexing='xy'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    # note the nearest neighbour interpolation method
    # note xy indexing because Guha's original code switched x and y dimensions
    nn_output = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)


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
