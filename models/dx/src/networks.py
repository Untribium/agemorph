"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, add
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Softmax
import tensorflow as tf


def classifier_net(vol_shape, layers, batchnorm=False, leaky=0.0, maxpool=False):

    vol_shape = tuple(vol_shape)

    ndims = len(vol_shape)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: {}".format(ndims)

    print('classifier net:')
    print('n_layers: {}'.format(len(layers)+1))

    x = Input(shape=vol_shape + (1,))
    print(K.int_shape(x))
    
    inputs = [x]

    for channels, strides in layers:
        if maxpool:
            x = conv_block(x, channels, batchnorm=batchnorm, leaky=leaky)
            print(K.int_shape(x))
            if strides > 1:
                x = maxpool_block(x, strides=strides, pool_size=strides)
        else:
            x = conv_block(x, channels, strides, batchnorm=batchnorm, leaky=leaky)
            print(K.int_shape(x))

    # add final (real) conv layer without batchnorm
    x = conv_block(x, layers[-1][0], batchnorm=False, leaky=leaky)
    print(K.int_shape(x))
 
    x = conv_block(x, 1, kernel_size=1, activation=False)
    print(K.int_shape(x))

    x = KL.Flatten()(x)
    print(K.int_shape(x))

    # weighted sum
    x = KL.Dense(2, use_bias=False)(x)
    print(K.int_shape(x))
    
    x = Softmax()(x)

    outputs = [x]
 
    return Model(inputs=inputs, outputs=outputs)


# Helper functions
def conv_block(x_in, nf, strides=1, kernel_size=3, activation=True, batchnorm=False, leaky=0.0):
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
