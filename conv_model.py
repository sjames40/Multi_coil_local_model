import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import numpy as np
import glob
import os
import cv2
import xarray as xr
import logging
import skimage
from skimage import transform
import src.data
import src.models
import src.descatter
import src.experiments
import src.reports as reports
from abc import ABC, abstractmethod
import scipy.interpolate
import os
import matplotlib.pyplot as plt
import torch
import itertools as it
import pandas as pd
import logging
import xarray as xr

data_name = 'multi'
model_type = 'local_conv'
#data_name2 = 'mono'
data_name_2 = 'multi_2'
data_name_3 = 'multi_3'
data_name_4 = 'multi_4'
# set up a log file
# src.experiments.setup_logging('test')

# load image daota
# rho, D, S, T, test, train = src.data.prep(data_name)
rho, D, S, T, test, train = src.data.prep(data_name)

rho1, D1, S1, T1, _, _ = src.data.prep(data_name_2)
rho2, D2, S2, T2, _, _ = src.data.prep(data_name_3)
rho3, D3, S3, T3, _, _ = src.data.prep(data_name_4)
total_data_clean = np.zeros((490, 257, 257))
total_data_clean[0:490, :, :] = S.sel(train).data
# total_data_clean[490:990,:,:] = S1.data
# total_data_clean[990:1490,:,:] = S2.data
# total_data_clean[1490:1890,:,:] = S3.data[0:400,:,:]
clean_data = total_data_clean
total_data_noisy = np.zeros((490, 257, 257))
#total_data_density = np.zeros((490, 257, 257))
total_data_noisy[0:490, :, :] = D.sel(train).data
#total_data_density[0:490, :, :] = rho.sel(train).data
#total_data_transmission = np.zeros((490, 257, 257))
#total_data_transmission[0:490, :, :] = T.sel(train).data

# total_data_noisy[490:990,:,:] = D1.data
# total_data_noisy[990:1490,:,:] = D2.data
# total_data_noisy[1490:1890,:,:] = D3.data[0:400,:,:]
noisy_data = total_data_noisy
def to_3d_array(x):
    """
    return x, reshaped (if possible) to be
    a num_images by rows by cols array

    does not copy!
    """

    if type(x) is xr.DataArray:
        x = x.data

    if x.ndim == 2:
        x = x[np.newaxis]

    if x.ndim != 3:
        raise ValueError('x should be (num_images, rows, cols) '
                         f'or rows x cols, but was {x.shape}')

    return x


def downsample(x, factor):
    """
    downsamples the images in x by factor in rows and cols
    via block averaging (ignoring nans, aka np.nanmean)


    x: (..., rows, cols) np.Array (stack of) image(s)
    """


    # pad to the next multiple of factor in each dim
    full_shape = np.array(x.shape)
    im_shape = full_shape[-2:]  # last two dims
    im_pad_shape = ((im_shape-1) // factor + 1) * factor
    full_pad_shape = full_shape.copy()
    full_pad_shape[-2:] = im_pad_shape

    pad_width = list(  # pad needs a list of (before, after) pairs
        zip(np.zeros_like(full_shape), full_pad_shape - full_shape)
        )
    x = np.pad(x, pad_width, mode='edge')

    # perform block averaging
    # shape: (..., M, N)
    x = x.reshape(*x.shape[:-2], factor, -1, x.shape[-1], order='f')
    # shape: (..., f, M/f, N)
    x = x.reshape(*x.shape[:-1], -1, factor)
    # shape: (..., f, M/f, N/f, f)

    counts = np.sum(~np.isnan(x), axis=(-4, -1))
    all_nan = counts == 0
    sums = np.nansum(x, axis=(-4, -1))
    counts[counts == 0] = 1

    means = sums / counts
    means[all_nan] = np.nan

    return means


def upsample(im, factor):
    def interp1d(im, axis):
        if axis == -1:
            fill_val = (im[..., :, 0], im[..., :, -1])
        elif axis == -2:
            fill_val = (im[..., 0, :], im[..., -1, :])
        x = np.arange(im.shape[axis])*factor + factor/2 - 0.5
        f = scipy.interpolate.interp1d(
            x, im, axis=axis, fill_value=fill_val, bounds_error=False)

        x_up = np.arange(im.shape[axis] * factor)
        return f(x_up)

    return interp1d(interp1d(im, -1), -2)


# print(clean_data[1])
# clean_ref = S.sel(test).data[6]
# noisy_ref = D.sel(test).data[6]
# q_from = np.nan_to_num(q_from, nan=0.0, copy=True)
# q_to = np.nan_to_num(q_to, nan=0.0, copy=True)
class ScatterModel(ABC):
    @abstractmethod
    def fit(self, D, S):
        """
        fit a model from D to S

        D: a num_images by rows by cols array, e.g., direct
        S: a num_images by rows by cols array, e.g., scatter
            must be the same shape as D

        returns: Dict of fitting results or None
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, D, D_train, S_train, T):
        """
        evaluate the model on each image in D to estimate S
        usually called after `fit`ing a training set

        D: a num_images by rows by cols array, e.g., direct
        D_train: a num_images by rows by cols array
            part of a training set of D, S pairs, used by local
            models
        S_train: a num_images by rows by cols array
        T: A num_images by rows by cols array,
            the transmission corresponding to D. Must be the
            same size as D. Not used by all models.

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def make_reports(self, out_dir, suffix=None):
        pass

    def prep(self, x):
        """
        make sure x is shape (N, row, col) and
        downsample if required.

        usually run at the start of fit()
        """
        self.in_shape = x.shape

        x = to_3d_array(x)

        if self.downrate > 1:
            x = downsample(x, self.downrate)
        else:
            x = x.copy()

        return x

    def deprep(self, y):
        """
        undo the steps in prep, so that if the function
        was called with a (row, col) array, it returns a (row, col)
        array instead of a (1, row, col) one

        also upsamples

        usually called at the end of fit
        """

        # return to original shape
        if self.downrate > 1:
            y = upsample(y, self.downrate)
            # crop to original shape
            y = y[..., :self.in_shape[-2], :self.in_shape[-1]]

        return y.reshape(self.in_shape)  # reshape may remove leading (1,)

def prep(x,downrate):
    """
    make sure x is shape (N, row, col) and
    downsample if required.

    usually run at the start of fit()
    """
    in_shape = x.shape

    x = to_3d_array(x)

    if downrate > 1:
        x = downsample(x, downrate)
    else:
        x = x.copy()

    return x
def norm(x, y):
    return np.nanmean(np.power(x - y, 2), axis=(1, 2))

class ConvModel(ScatterModel):
    """
    S(r) = (h*D)(r)
    """
    def __init__(self, kernel_size, h=None, h0=None,
                 rho=1.0, num_iters=10, num_inner_iters=40, downrate=1,
                 nonnegative=True):
        self.kernel_size = kernel_size
        self.rho = rho
        self.h = h
        self.h0 = h0
        self.num_iters = num_iters
        self.num_inner_iters = num_inner_iters
        self.downrate = downrate
        self.nonnegative = nonnegative

    def fit(self, x, y):
        x, y = map(self.prep, (x, y))

        is_missing = np.isnan(y)
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0

        self.x_norm, self.y_norm = (src.algo.nannorm(q) for q in (x, y))

        # normalize the data
        # this is really useful for setting rho in ADMM
        # however subtracting means is dangerous because
        # some models are linear, not affine
        # x -= self.x_mean
        # y -= self.y_mean
        x = x / self.x_norm
        y = y / self.y_norm
        print(y.shape)
        def A(w):
            y = src.algo.conv2dfft(x, w)
            y[is_missing] = 0
            return y

        HxT = src.algo.make_conv_by_x_T(x, self.kernel_size)

        def AT(q):
            q[is_missing] = 0
            return HxT(q)

        if self.nonnegative:
            w, dual, log = src.algo.solve_NNLS(
                y, A, AT, x0=self.h0,
                rho=self.rho, num_iters=self.num_iters,
                num_inner_iters=self.num_inner_iters, verbose=False)

        else:
            def ATA(w):
                return AT(A(w))
            w = src.algo.conj_grad(ATA, AT(y), num_iters=40, verbose=True)
            log = None

        self.h = w.copy()
        if self.nonnegative:
            self.h[self.h < 0] = 0

        self.h_negative = w

        return log

    def evaluate(self, x, *_unused):
        x = self.prep(x)

        is_missing = np.isnan(x)
        x[is_missing] = 0
        x = x / self.x_norm

        y = src.algo.conv2dfft(x, self.h)
        y = y * self.y_norm
        
        #np.save(os.path.join('..','descattering','training_losses','local_conv_1iteration.npy'),self.deprep(y))
        return self.deprep(y)

h_train = np.load(os.path.join('..', 'descattering', 'x_kernel.npz'))
h_test = np.load(os.path.join('..', 'descattering', 'x_kernel_test.npz'))


list_inds = []
num_neighbors = 5
total_data_kernel = np.zeros((490, 257, 257))
for j in range(490):
    image_kernel = D.sel(test).data[0]  # image_
    image_kernel = prep(image_kernel, 1)
    image_kernel[np.isnan(image_kernel)] = 0
    image_kernel = src.algo.conv2dfft(image_kernel, h_train['arr_0'][j])
    total_data_kernel[j, :, :] = image_kernel
for i in range(1):
    # noisy_ref = D.sel(test).data[i]
    # noisy_ref_noisy = rho.sel(test).data[3]
    # noisy_ref = np.nan_to_num(noisy_ref, nan=0.0, copy=True)
    # noisy_ref = np.log(noisy_ref)
    # noisy_ref_noisy = np.load(os.path.join('..','descattering','reports','x_local_2layer_val_RC_D_to_S_noisy_inds1.npy'))
    # noisy_ref_noisy = np.nan_to_num(noisy_ref_noisy, nan=0.0, copy=True)
    # noisy_ref_noisy = np.log(noisy_ref_noisy)
    # noisy_data = np.nan_to_num(noisy_data, nan=0.0, copy=True)
    # total_data_transmission = np.nan_to_num(total_data_transmission, nan=0.0, copy=True)
    # total_data_density = np.nan_to_num(total_data_density, nan=0.0, copy=True)
    # noisy_data = np.log(noisy_data)
    # distances = norm(noisy_ref, noisy_data)
    noisy_ref = D.sel(test).data[0]
    clean_ref = S.sel(test).data[0]
    noisy_ref = np.nan_to_num(noisy_ref, nan=0.0, copy=True) 
    clean_ref = np.nan_to_num(clean_ref, nan=0.0, copy=True)
    distances_kernel = norm(clean_ref, total_data_kernel)

    # distances_profile = norm(noisy_ref_noisy, total_data_density)
    # distances_transmission = norm(noisy_ref_noisy, total_data_transmission)
    # distances_clean = norm(noisy_ref, noisy_data)
    inds = np.argsort(distances_kernel)#[:num_neighbors]
    # print(inds)



    # list_inds.append(accuracy1)
    x_train_local = noisy_data[inds[:num_neighbors]]
    y_train_local = clean_data[inds[:num_neighbors]]
    print(x_train_local.shape)
    x_train, y_train = (to_3d_array(v) for v in
                               (x_train_local, y_train_local))
    model = ConvModel(kernel_size=129,
            rho=1.0e0,
            num_iters=40,
            num_inner_iters=10,
            downrate=4)
    model.fit(x_train, y_train)

    result = model(noisy_ref)
    result = np.nan_to_num(result, nan=0.0, copy=True)
    MSE_one =np.linalg.norm(result-clean_ref)/np.linalg.norm(clean_ref)
    MSE_one = np.round(MSE_one,5)
    #error = reports.NMSE_percent(result.astype(np.float32), clean_ref.astype(np.float32))
    print(MSE_one)

    
    
    
    
    
    np.save(os.path.join('..','descattering','reports','conv_one_layer_recon.npy'),result)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    

    


