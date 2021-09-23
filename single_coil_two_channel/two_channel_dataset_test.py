import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'notebook')
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import time
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import scipy.io as sio
from ismrmrdtools import show, transform
# import ReadWrapper
import read_ocmr as read
import h5py
import sys
import os
import torch
from torch.utils.data.dataset import Dataset
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2,ifft2_mask, cplx_to_tensor, complex_conj, complex_matmul, absolute
import h5py




#arr = os.listdir(rt)
count =0

kspace_data_name = '/home/liangs16/MRI_descattering/RC_Image_4_fold_2channel/'
kspace_data = []
kspace_array = os.listdir(kspace_data_name)
print(len(kspace_array)*0.95)


def center_crop(data, shape):

    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def make_vdrs_mask(N1,N2,nlines,init_lines):
    # Setting Variable Density Random Sampling (VDRS) mask
    mask_vdrs=np.zeros((N1,N2),dtype='bool')
    mask_vdrs[:,int(0.5*(N2-init_lines)):int(0.5*(N2+init_lines))]=True
    nlinesout=int(0.5*(nlines-init_lines))
    rng = np.random.default_rng()
    t1 = rng.choice(int(0.5*(N2-init_lines))-1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(int(0.5*(N2+init_lines))+1, N2), size=nlinesout, replace=False)
    mask_vdrs[:,t1]=True; mask_vdrs[:,t2]=True
    return mask_vdrs

def convert_2chan_into_abs(img): # to complex
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    #show_img(torch.abs(img_complex).cpu(), cmap='gray')
    return img_complex



class nyumultidataset(Dataset):
    def __init__(self,data):        
        self.A_paths = data
       # self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.nx = 640
        self.ny = 322
        #self.mask =make_vdrs_mask(640,322,54,26)
        #if (opt.mask_type == 'mat'):
        #    self.mask = np.load(opt.mask_path)

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
       # A_temp = h5py.File(A_path, 'r')
        k_r = A_temp['k_r'][()] # real channel part of image
        k_i = A_temp['k_i'][()]
        nx,ny = k_r.shape
        mask = make_vdrs_mask(nx,ny,np.int(ny*0.25),np.int(ny*0.06)) # 4 acceleration
        k_r = np.reshape(k_r, [ nx, ny],order='A')
        k_i = np.reshape(k_i, [ nx, ny],order='A')
        k_np = np.stack((k_r, k_i), axis=0)
        mask_two_channel = np.stack((mask,mask),axis=0)
        A_k = torch.tensor(k_np, dtype=torch.float32)
        A_I = ifft2(A_k.permute(1,2,0)).permute(2,0,1)
        A_k_mask =A_k
        A_k_mask[~mask_two_channel, :] = 0
        A_I_mask =ifft2(A_k_mask.permute(1,2,0)).permute(2,0,1)
        #A_I = A_I[:, nx//2-nx//2:nx//2+nx//2,ny//2-ny//2:ny//2+ny//2]
        #A_I_mask = A_I_mask[:, nx//2-nx//2:nx//2+nx//2,ny//2-ny//2:ny//2+ny//2]
        A_I_crop = center_crop(A_I, (320,320))
        A_I_mask_crop = center_crop(A_I_mask, (320,320))
        mask = center_crop(mask,(320,320))
        return  A_I_mask_crop, A_I_crop,mask
    def __len__(self):
        return len(self.A_paths)
train_size = 0.95
len_data = len(kspace_data) 
num_train = int(len_data*train_size) 
train_dataset = nyumultidataset(clean_data1)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,shuffle=True)

#for direct, target in train_loader:
    #np.save(os.path.join('..','MRI_descattering','Test_image','1181_simliar_neig_2channel.npy'),direct.detach().numpy())
#    np.save(os.path.join('..','MRI_descattering','Test_image','1181_similiar_neigh_2channel.npy'),target.detach().numpy())

    
    

    
    
    
    
    
    
    
    
    




    






    
