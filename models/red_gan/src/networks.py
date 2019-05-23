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
from keras.layers import Conv3D, Activation, Input, UpSampling3D
from keras.layers import concatenate, add, subtract
from keras.layers import ReLU, Reshape, Lambda, BatchNormalization
from keras.losses import sparse_categorical_crossentropy
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
               reg_model_file, clf_model_file, batchnorm, leaky):

    vol_shape = tuple(vol_shape)
   
    gen_net = generator_net(vol_shape, vel_resize, int_steps, batchnorm, leaky)
    cri_net = critic_net(vol_shape, cri_base_nf, leaky, reg_model_file is None)

    """
    nomenclature
    x = img 1st visit
    y = img 2nd visit
    r = real
    f = fake
    a = avg
    D = critic output
    """
    
    # --- age regressor (pre-trained) ---
    if reg_model_file:
        reg_net = keras.models.load_model(reg_model_file)
        reg_net.name = 'reg'
        reg_net.trainable = False
 
    # --- dx classifier (pre-trained) ---
    if clf_model_file:
        clf_net = keras.models.load_model(clf_model_file)
        clf_net.name = 'clf'
        clf_net.trainable = False

    # --- critic ---

    gen_net.trainable = False # freeze generator in critic

    # critic inputs
    xr_cri = Input(shape=vol_shape + (1,)) # real 1st visit
    yr_cri = Input(shape=vol_shape + (1,)) # real 2nd visit
    dt_cri_bin = Input(shape=(16,))  # delta (binary)

    cri_inputs = [xr_cri, yr_cri, dt_cri_bin]

    # generate image using generator
    gen_out = gen_net([xr_cri, dt_cri_bin])
    yf_cri = gen_out[0]

    # interpolate sample for gradient penalty
    ya_cri = RandomWeightedAverage(batch_size=batch_size)([yr_cri, yf_cri])

    # critic inputs for real, fake and interpolated
    cri_in_r = [xr_cri, yr_cri]
    cri_in_f = [xr_cri, yf_cri]
    cri_in_a = [xr_cri, ya_cri]
    
    # add delta channel to critic inputs if we're not using an age regressor
    if not reg_model_file:
        dt_cri_cnl = Input(shape=vol_shape + (1,))  # delta (channel)
        
        cri_inputs.append(dt_cri_cnl)

        cri_in_r.append(dt_cri_cnl)
        cri_in_f.append(dt_cri_cnl)
        cri_in_a.append(dt_cri_cnl)
    
    Dr_cri = cri_net(cri_in_r)
    Df_cri = cri_net(cri_in_f)
    Da_cri = cri_net(cri_in_a)

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

    xr_gen = Input(shape=vol_shape + (1,)) # real 1st visit
    yr_gen = Input(shape=vol_shape + (1,)) # real 2nd visit
    dt_gen_bin = Input(shape=(16,))  # delta (binary)
    
    gen_inputs = [xr_gen, yr_gen, dt_gen_bin]

    # predict y_hat
    gen_out = gen_net([xr_gen, dt_gen_bin])
    yf_gen = gen_out[0]

    # critic inputs
    cri_in_f = [xr_gen, yf_gen]

    # add delta channel to critic inputs if we're not using an age regressor
    if not reg_model_file:
        dt_gen_cnl = Input(shape=vol_shape + (1,))  # delta (channel)
        gen_inputs.append(dt_gen_cnl)

        cri_in_f.append(dt_gen_cnl)

    # get wasserstein loss
    Df_gen = cri_net(cri_in_f)

    gen_outputs = [Df_gen, *gen_out]
    
    gen_loss = [loss_class.wasserstein_loss,
                loss_class.l1_loss,
                loss_class.kl_loss,
                loss_class.l1_loss,
                loss_class.dummy_loss,
                loss_class.dummy_loss]

    # calculate age regressor delta and add to outputs
    if reg_model_file:
        reg_yr_gen = reg_net([yr_gen])
        reg_yf_gen = reg_net([yf_gen])
        reg_dt_gen = subtract([reg_yf_gen, reg_yr_gen])

        gen_outputs.append(reg_dt_gen)
        gen_loss.append(loss_class.l1_loss)
    
    # calculate dx classifier logits and add to outputs 
    if clf_model_file:
        clf_yr_gen = clf_net([yr_gen])
        clf_yf_gen = clf_net([yf_gen]) # dx logits [mci, ad]
        
        # soft-label cross entropy loss
        clf_xe_gen = CrossEntropy()([clf_yr_gen, clf_yf_gen]) 

        print(K.int_shape(clf_xe_gen))

        gen_outputs.append(clf_xe_gen)
        gen_loss.append(loss_class.l1_loss)
 
    gen_model = Model(inputs=gen_inputs, outputs=gen_outputs)

    gen_model.compile(loss=gen_loss, optimizer=gen_optimizer, loss_weights=gen_loss_weights)

    return cri_model, gen_model


def unet_core(vol_shape, vel_resize, batchnorm, leaky):
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
   
    shapes = []
    for i in range(5):
        shape = np.array(vol_shape) // 2**i
        shapes.append(tuple(shape))
 
    upsample_layer = getattr(KL, 'UpSampling{}D'.format(ndims))

    print('generator unet:')

    # inputs
    e0 = Input(shape=vol_shape + (1,))
    print(K.int_shape(e0))

    inputs = [e0]

    e1 = conv_block(e0,  8, 1, batchnorm=batchnorm, leaky=leaky) # 80 32 80   8
    print(K.int_shape(e1))
    e2 = conv_block(e1, 16, 2, batchnorm=batchnorm, leaky=leaky) # 40 16 40  16
    print(K.int_shape(e2))
    e3 = conv_block(e2, 16, 1, batchnorm=batchnorm, leaky=leaky) # 40 16 40  16
    print(K.int_shape(e3))
    e4 = conv_block(e3, 32, 2, batchnorm=batchnorm, leaky=leaky) # 20  8 20  32
    print(K.int_shape(e4))
    e5 = conv_block(e4, 32, 1, batchnorm=batchnorm, leaky=leaky) # 20  8 20  32
    print(K.int_shape(e5))
    e6 = conv_block(e5, 64, 2, batchnorm=batchnorm, leaky=leaky) # 10  4 10  64
    print(K.int_shape(e6))
    e7 = conv_block(e6, 64, 1, batchnorm=batchnorm, leaky=leaky) # 10  4 10  64
    print(K.int_shape(e7))
    e8 = conv_block(e7, 64, 2, batchnorm=batchnorm, leaky=leaky) #  5  2  5  64
    print(K.int_shape(e8))

    e9 = KL.Flatten()(e8)
    print(K.int_shape(e9))

    features = KL.Dense(400)(e9)
    print(K.int_shape(features))

    n_units = np.array(shapes[-1]).prod() * 64
    d9 = KL.Dense(n_units)(features)
    print(K.int_shape(d9))

    d8 = KL.Reshape((*shapes[-1], 64))(d9)                       #  5  2  5  64
    d8 = concatenate([e8, d8])                                   #  5  2  5 128
    d8 = conv_block(d8, 64, 1, batchnorm=batchnorm, leaky=leaky) #  5  2  5  64
    print(K.int_shape(d8))

    d7 = upsample_layer()(d8)                                    # 10  4 10  64
    d7 = conv_block(d7, 64, 1, batchnorm=batchnorm, leaky=leaky) # 10  4 10  64
    print(K.int_shape(d7))
    d7 = concatenate([e7, d7])                                   # 10  4 10 128
    d6 = conv_block(d7, 64, 1, batchnorm=batchnorm, leaky=leaky) # 10  4 10  64
    print(K.int_shape(d6))

    d5 = upsample_layer()(d6)                                    # 20  8 20  64
    d5 = conv_block(d5, 32, 1, batchnorm=batchnorm, leaky=leaky) # 20  8 20  32
    print(K.int_shape(d5))
    d5 = concatenate([e5, d5])                                   # 20  8 20  64
    d4 = conv_block(d5, 32, 1, batchnorm=batchnorm, leaky=leaky) # 20  8 20  32
    print(K.int_shape(d4))

    d3 = upsample_layer()(d4)                                    # 40 16 40  32
    d3 = conv_block(d3, 16, 1, batchnorm=batchnorm, leaky=leaky) # 40 16 40  16
    print(K.int_shape(d3))
    d3 = concatenate([e3, d3])                                   # 40 16 40  32
    d2 = conv_block(d3,  8, 1, batchnorm=False,     leaky=leaky) # 40 16 40   8
    print(K.int_shape(d2))

    if vel_resize == 1.0:
        d1 = upsample_layer()(d2)                                # 80 32 80   8
        print(K.int_shape(d1))
        d1 = concatenate([e1, d1])                               # 80 32 80  16
        d0 = conv_block(d1, 3, 1, batchnorm=False,  leaky=leaky) # 80 32 80   3
        print(K.int_shape(d0))
        
        outputs = [d0, features]
    else:
        outputs = [d2, features]
   
    return Model(inputs=inputs, outputs=outputs)


def generator_net(vol_shape, vel_resize, int_steps, batchnorm, leaky):
 
    ndims = len(vol_shape)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)
    
    unet = unet_core(vol_shape, vel_resize, batchnorm, leaky)
    
    # target delta in binary representation
    b_in = Input(shape=(16,)) 
    
    x_in = unet.inputs[0]
    x_out, features = unet.outputs

    inputs = [x_in, b_in] 
    
    Conv = getattr(KL, 'Conv{}D'.format(ndims))

    flow_mean = Conv(ndims, kernel_size=3, padding='same', name='flow_mean',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x_out)

    flow_log_sigma = Conv(ndims, kernel_size=3, padding='same', name='flow_log_sigma',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                bias_initializer=keras.initializers.Constant(value=-10))(x_out)

    flow_params = concatenate([flow_mean, flow_log_sigma])
    
    flow_magnitude = Magnitude()(flow_mean)

    # sample velocity field (using reparameterization)
    z_sample = Sample(name='z_sample')([flow_mean, flow_log_sigma])

    # integrate flow
    flow = VecInt(method='ss', name='flow_int', int_steps=int_steps)([z_sample, b_in])

    # resize to full size 
    if vel_resize != 1.0:
        flow = trf_resize(flow, vel_resize, name='flow_resize')

    y = SpatialTransformer(interp_method='linear', indexing='ij')([x_in, flow])
  
    outputs = [y, flow_params, flow_magnitude, flow, features]

    return Model(inputs=inputs, outputs=outputs)


def critic_net(vol_shape, base_nf=8, leaky=0.2, delta_in=False):

    ndims = len(vol_shape)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    x = Input(shape=vol_shape + (1,)) # image 1st visit
    y = Input(shape=vol_shape + (1,)) # image 2nd visit (real or fake)

    inputs = [x, y]

    if delta_in:
        d = Input(shape=vol_shape + (1,))
        inputs.append(d)

    print('critic net:')

    l = concatenate(inputs)
    print(K.int_shape(l))

    l = conv_block(l, base_nf,    1, leaky=leaky)
    print(K.int_shape(l))

    l = conv_block(l, base_nf*2,  2, leaky=leaky)
    print(K.int_shape(l))
    l = conv_block(l, base_nf*2,  1, leaky=leaky)
    print(K.int_shape(l))
   
    l = conv_block(l, base_nf*4,  2, leaky=leaky)
    print(K.int_shape(l))
    l = conv_block(l, base_nf*4,  1, leaky=leaky)
    print(K.int_shape(l))
    
    l = conv_block(l, base_nf*8,  2, leaky=leaky)
    print(K.int_shape(l))
    l = conv_block(l, base_nf*8,  1, leaky=leaky)
    print(K.int_shape(l))
    
    l = conv_block(l, base_nf*16, 1, leaky=leaky) # stride 1!
    print(K.int_shape(l))
    l = conv_block(l, base_nf*16, 1, leaky=leaky)
    print(K.int_shape(l))
    
    l = conv_block(l, 1, kernel_size=1, activation=False)
    print(K.int_shape(l))

    l = KL.Flatten()(l)
    print(K.int_shape(l))

    # weighted sum
    l = KL.Dense(1, use_bias=False)(l)
    print(K.int_shape(l))
    
    outputs = [l]
 
    return Model(inputs=inputs, outputs=outputs)


# Helper functions
def conv_block(x_in, nf, strides=1, kernel_size=3, activation=True, batchnorm=False, leaky=0.2):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    Conv = getattr(KL, 'Conv{}D'.format(ndims))
    
    x_out = Conv(nf, kernel_size=kernel_size, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    
    if activation:
        x_out = ReLU(negative_slope=leaky)(x_out)
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


class Magnitude(Layer):
    """
    Keras Layer: calculate magnitude of flow
    """

    def __init__(self, **kwargs):
        super(Magnitude, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Magnitude, self).build(input_shape)

    def call(self, x):
        return K.sqrt(K.sum(x * x, axis=-1, keepdims=True))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


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


class CrossEntropy(Layer):
    
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CrossEntropy, self).build(input_shape)

    def call(self, x):
        return K.categorical_crossentropy(x[0], x[1], from_logits=True)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (1,)


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
