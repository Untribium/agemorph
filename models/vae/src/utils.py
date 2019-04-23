import tensorflow as tf
import numpy as np


def to_int(bin_digits):
    bin_string = ''.join(map(str, bin_digits[::-1]))
    value = int(bin_string, 2)
    return value


def to_bin(value, n_bits=16):
    bin_string = np.binary_repr(value, n_bits)[::-1]
    bin_digits = [int(d) for d in bin_string]
    return bin_digits


def convert_delta(gen, max_delta, int_steps, kl_dummy):
    
    while True:
        
        imgs, lbls = next(gen)

        lbls[0] = lbls[0][:, 0]

        delta = lbls[0] / (max_delta + 1)

        delta_shift = delta * 2**(int_steps + 1)
        delta_shift = delta_shift.astype(int)

        delta_bin = [to_bin(d, 16) for d in delta_shift]
        delta_bin = np.array(delta_bin)

        yield [imgs[0], delta_bin, *lbls[1:]], [imgs[1], kl_dummy]


def normalize_dim(tensor, axis):
    ndims = len(tensor.shape)

    axes = np.arange(ndims)
    axes = np.delete(axes, axis)

    maxs = np.abs(tensor).max(axis=tuple(axes))
    maxs[maxs == 0] = 1 # avoid div by 0

    slicer = [None] * ndims
    slicer[axis] = slice(None)

    tensor /= maxs[tuple(slicer)]

    return tensor
