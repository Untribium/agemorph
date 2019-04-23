"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os, sys
import numpy as np
import pandas as pd
import nibabel as nib
from .utils import to_bin


def vae_gen(csv_gen, max_delta, int_steps, kl_dummy):
    
    while True:
        
        imgs, lbls = next(csv_gen)

        lbls[0] = lbls[0][:, 0]

        delta = lbls[0] / (max_delta + 1)

        delta_shift = delta * 2**(int_steps + 1)
        delta_shift = delta_shift.astype(int)

        delta_bin = [to_bin(d, 16) for d in delta_shift]
        delta_bin = np.array(delta_bin)

        yield [imgs[0], delta_bin, *lbls[1:]], [imgs[1], kl_dummy]


def csv_gen(csv_path, img_keys, lbl_keys, batch_size, split=None, sample=True, 
                    shuffle=True, weights=None, n_epochs=None, verbose=False):
    """
    batch generator from csv
    Arguments:
    csv_path:   str
                absolute path to csv file
    img_keys:   list of str
                columns in csv containing nifti paths
    lbl_keys:   list of str
                columns in csv containing non-image data
    batch_size: int
                desired batch size
    split:      str
                which data split to use, expects 'split' column in csv
    sample:     bool
                sample each batch from all samples, otherwise shuffle then split
    shuffle:    bool
                shuffle csv before epoch (sample=False only)
    weights:    str
                sample weights columns (sample=True only)
    n_epochs:   int
                how many epochs to generate
    verbose:    bool
                print extra information while running

    """

    assert os.path.isfile(csv_path), 'csv not found at {}'.format(csv_path)

    csv = pd.read_csv(csv_path)

    if split is not None:
        assert 'split' in csv.columns, 'csv has no split column'
        csv = csv[csv['split'] == split]
    else:
        split = 'data'

    for key in img_keys + lbl_keys:
        csv = csv[~csv[key].isna()]

    csv.reset_index(inplace=True, drop=True)

    n_rows = csv.shape[0]

    print('generating batches from {} {} samples'.format(n_rows, split))

    n_batches = n_rows // batch_size # number of batches per epoch

    epoch = 0 # epoch count

    while n_epochs is None or epoch < n_epochs:

        if verbose:
            print('starting {} epoch {}'.format(split, epoch))

        if not sample and shuffle:
            csv = csv.sample(frac=1) # shuffle csv
            csv.reset_index(inplace=True, drop=True)
       
        for b in range(n_batches):

            if sample:
                batch = csv.sample(batch_size, weights=weights)
            else:
                batch = csv.iloc[b*batch_size:(b+1)*batch_size]
 
            # MR scans
            imgs = []

            for img_key in img_keys:
                concat = []
                
                for _, row in batch.iterrows():
                    img = load_nifti(row[img_key])
                    img = img[np.newaxis, ..., np.newaxis] # add batch_dim and channel_dim
                    concat.append(img)

                imgs.append(np.concatenate(concat, axis=0))

            # lbls (e.g. pat_dx, pat_age, img_id)
            lbls = []

            for lbl_key in lbl_keys:
                concat = []

                for _, row in batch.iterrows():
                    lbl = np.array([row[lbl_key]])
                    lbl = lbl[np.newaxis, ...] # add batch_dim
                    concat.append(lbl)

                lbls.append(np.concatenate(concat, axis=0))

            yield imgs, lbls

        epoch += 1

 
def load_nifti(path):
    """
    get numpy array from path to nifti file
    """
    
    nii = nib.load(path)
    vol = np.squeeze(nii.get_data().astype(np.float32))

    return vol



"""
original voxelmorph code
"""

def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def cvpr2018_gen_s2s(gen, batch_size=1):
    """ generator used for cvpr 2018 model for subject 2 subject registration """
    zeros = None
    while True:
        X1 = next(gen)[0]
        X2 = next(gen)[0]

        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        yield ([X1, X2], [X2, zeros])


def miccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if bidir:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def miccai2018_gen_s2s(gen, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    zeros = None
    while True:
        X = next(gen)[0]
        Y = next(gen)[0]
        if zeros is None:
            volshape = X.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros])


def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx])
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                X_seg = load_volfile(vol_names[idx].replace('norm', 'aseg'))
                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)
            
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        yield tuple(return_vals)

def load_example_by_name(vol_name, seg_name):
    """
    load a specific volume and segmentation
    """
    X = load_volfile(vol_name)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(seg_name)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data' 
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        X = np.load(datafile)['vol_data']

    return X
