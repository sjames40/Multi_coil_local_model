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
from torch.utils.data.dataset import Dataset
from torch.nn import init
import math
import scipy
import scipy.linalg
import read_ocmr as read
import h5py
import sys
import os
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
import h5py



#rt = '/home/liangs16/shared_data/mulitcoil_half'
kspace_data_name = '/home/shijunliang/MRI_sampling/NEW_8_fold'

kspace_data = []
kspace_array = os.listdir(kspace_data_name)
print(len(kspace_array))

#for i in range(len(kspace_array)):
for i in range(len(kspace_array)): 
    kspace_file = kspace_array[i]
    kspace_data_from_file = np.load(os.path.join(kspace_data_name,kspace_file),'r')
    kspace_data.append(kspace_data_from_file)

#arr = os.listdir(rt)
#count =0
def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
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




#$mask = make_vdrs_mask(640,400,54,26) # 4 accelration mask
#mask = make_vdrs_mask(640,400,54,26)

class nyumultidataset(Dataset):
    def  __init__(self ,kspace_data):
        self.A_paths = kspace_data
       # self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

        self.nx = 640
        self.ny = 400
       # self.mask = make_vdrs_mask(640,400,np.int(self.ny*0.07),np.int(self.ny*0.03))

       # if (opt.mask_type == 'mat'):
       #     self.mask = np.load(opt.mask_path)

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        s_r = A_temp['s_r']/ 32767.0 
        s_i = A_temp['s_i']/ 32767.0 
        k_r = A_temp['k_r']/ 32767.0
        k_i = A_temp['k_i']/ 32767.0 
        ncoil, nx, ny = s_r.shape
        mask = make_vdrs_mask(nx,ny,np.int(ny*0.07),np.int(ny*0.03))
        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        mask = np.stack((mask,mask),axis=0)
        mask =torch.tensor(mask)
        mask_two_channel = mask.repeat(ncoil, 1, 1, 1)#.permute(1, 0, 2, 3)
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        mask_two_channel =mask_two_channel.numpy()
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)), dim=0)
        A_I_crop = SOS / torch.max(torch.abs(SOS)[:])
       
        A_k_mask = A_k
        A_k_mask[~mask_two_channel, :] = 0
       
        A_I_mask =ifft2(A_k_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 
        #mask_two_channel = mask_two_channel[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        #A_s = A_s[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_I_mask = A_I_mask[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_s_mask = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS_mask = torch.sum(complex_matmul(A_I_mask, complex_conj(A_s_mask)), dim=0)
        # print('sossize',SOS.size())
        A_I_mask_crop = SOS_mask / torch.max(torch.abs(SOS_mask)[:])
        mask = mask[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        
        return  A_I_mask_crop, A_I_crop,A_s,mask
     
       
    def __len__(self):
        return len(self.A_paths)

len_data = len(kspace_data)


train_size = 0.9
num_train = int(len_data*train_size)
    
test_clean_paths = kspace_data[:num_train]
train_dataset = nyumultidataset(test_clean_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True)
#for direct, target in train_loader:
#    print(direct.shape)
#for direct,target in train_loader:
#    print(direct)
#    print(direct.shape) 

    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    # np.save(os.path.join('..','MRI_descattering','SOS_reconstruct.npy'),direct.detach().numpy())

    
    
    
    

    


    
    

    

    
    
    
    
