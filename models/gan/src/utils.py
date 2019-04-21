import numpy as np


def to_bin(value, n_bits=16):
    bin_string = np.binary_repr(value, n_bits)[::-1]
    bin_digits = [int(d) for d in bin_string]
    return bin_digits


def convert_delta(gen, max_delta, int_steps):
        '''
        convert delta into bin representation used in generator ss steps.
        number of bits is fixed to 16, each bit signifies whether that step
        is applied to the total displacement
        '''
        while(True):

            imgs, lbls = next(gen) 

            lbls[0] = lbls[0][:, 0] # squeeze last dim
 
            delta = lbls[0] / (max_delta + 1) # deltas in days
            
            delta_shift = delta *  2**(int_steps + 1)
            delta_shift = delta_shift.astype(int)

            delta_bin = [to_bin(d, 16) for d in delta_shift]

            lbls[1:1] = [np.array(delta_bin)]

            '''
            # binary representation
            scaled = (delta_norm * 255).astype(np.uint8).reshape((-1, 1))
            lbls[1:1] = [np.unpackbits(scaled, axis=1)[:, int_steps::-1]]
            '''
            
            # channel
            delta_channel = delta.reshape((-1, 1, 1, 1, 1))
            lbls[0] = np.ones_like(imgs[0], dtype=np.float32) * delta_channel

            yield imgs, lbls


def normalize(tensor, axis=0):
    """
    normalize each tensor along given axis to [-1, 1], e.g. each sample in batch
    """
    ndims = len(tensor.shape)

    axes = np.arange(ndims)
    axes = np.delete(axes, axis)

    maxs = np.abs(tensor).max(axis=tuple(axes))
    maxs[maxs == 0] = 1 # avoid div by 0

    slicer = [None] * ndims
    slicer[axis] = slice(None)

    tensor /= maxs[tuple(slicer)]

    return tensor
