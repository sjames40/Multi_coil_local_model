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
kspace_data_name = '/home/liangs16/labmat_project/MRI_descattering/NEW_8_fold'

kspace_data = []
kspace_array = os.listdir(kspace_data_name)
print(len(kspace_array))

#for i in range(1):
for i in range(len(kspace_array)): 
    kspace_file = kspace_array[i]
    kspace_data_from_file = np.load(os.path.join(kspace_data_name,kspace_file),'r')
    kspace_data.append(kspace_data_from_file)

#arr = os.listdir(rt)
#count =0
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


image_space_data =[]
image_space_gh =[]
'''
for j in range(len(kspace_data)):
    A_temp = self.A_paths[index]
    s_r = A_temp['s_r']/ 32767.0
    s_i = A_temp['s_i']/ 32767.0
    k_r = A_temp['k_r']/ 32767.0
    k_i = A_temp['k_i']/ 32767.0
    ncoil, nx, ny = s_r.shape
    mask = make_vdrs_mask(nx,ny,np.int(ny*0.14),np.int(ny*0.07))
    k_np = np.stack((k_r, k_i), axis=0)
    s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
    mask = np.stack((mask,mask),axis=0)
    mask =torch.tensor(mask)
    mask_two_channel = mask.repeat(ncoil, 1, 1, 1)#.permute(1, 0, 2, 3)
    A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
    mask_two_channel =mask_two_channel.numpy()
    A_k_mask = A_k
    A_k_mask[~mask_two_channel, :] = 0
        #A_k_mask =  torch.tensor(A_, dtype=torch.float32).permute(1, 0, 2, 3)
    A_I_mask =ifft2(A_k_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
    A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
    SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)), dim=0)
        #print('sossize',SOS.size())
    A_I = SOS / torch.max(torch.abs(SOS)[:])
    A_I_mask = A_I_mask[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        #A_s_mask = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
    SOS_mask = torch.sum(complex_matmul(A_I_mask, complex_conj(A_s)), dim=0)
        # print('sossize',SOS.size())
    A_I_mask = SOS_mask / torch.max(torch.abs(SOS_mask)[:])
    image_space_gh.append(A_I_mask)
#    np.save(os.path.join('/home/liangs16/MRI_descattering/GT_image_4_fold_2channel_image_domain','%2channel%d.npy'%(groundturth,j)),A_I)
#    np.save(os.path.join('/home/liangs16/MRI_descattering/RC_image_4_fold_2channel_image_domain','%2channel%d.npy'%(recon,j)),A_I_mask)
    
norm_matrix =[]
norm_matrix2 =[]
#len(kspace_array)
for a in range(len(kspace_array)):
   # norms = np.linalg.norm(torch.view_as_complex(image_space_gh[1182].permute(1,2,0))-torch.view_as_complex(image_space_gh[a].permute(1,2,0)),'fro')# norm matrix x is clean patch_image
    norms =np.linalg.norm(np.abs(image_space_gh[1181])-np.abs(image_space_gh[a]))
    #norms2 = torch.mean(torch.abs(torch.view_as_complex(image_space_gh[1100].permute(1,2,0))-torch.view_as_complex(image_space_gh[1].permute(1,2,0))))
    norm_matrix.append(norms)
#    norm_matrix2.append(norms)
match_inds = np.argsort(norm_matrix)[1:50+1]
#match_inds = np.argsort(norm_matrix2)[1:40+1]

match_inds.sort()
print(match_inds)



clean_data1 = []

for b in range(len(match_inds)):
    clean_data1.append(kspace_data[(match_inds[b])])


#$mask = make_vdrs_mask(640,400,54,26) # 4 accelration mask
#mask = make_vdrs_mask(640,400,54,26)
'''
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
        mask = make_vdrs_mask(nx,ny,np.int(ny*0.14),np.int(ny*0.07))
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
        #print('sossize',SOS.size())
        
        A_I_crop = SOS / torch.max(torch.abs(SOS)[:])
        A_k_mask = A_k
        A_k_mask[~mask_two_channel, :] = 0
        A_I_mask =ifft2(A_k_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I_mask = A_I_mask[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_s_mask = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS_mask = torch.sum(complex_matmul(A_I_mask, complex_conj(A_s_mask)), dim=0)
        # print('sossize',SOS.size())
        A_I_mask_crop = SOS_mask / torch.max(torch.abs(SOS_mask)[:])
        return A_I_mask_crop,A_I_crop
     
       
    def __len__(self):
        return len(self.A_paths)

len_data = len(kspace_data)
#print(len_data)

train_size = 0.9
num_train = int(len_data*train_size)
    
train_clean_paths = kspace_data[:num_train]
train_dataset = nyumultidataset(train_clean_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True)

test_clean_paths = kspace_data[num_train:num_train+1]
test_dataset = nyumultidataset(test_clean_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)



#for direct, target in train_loader:
#    print(direct.shape)
#for direct,target in train_loader:
    #print(direct)
    #print(direct.shape) 
#    np.save(os.path.join('..','MRI_descattering','two_channel_result','SOS_noise.npy'),direct.detach().numpy())
#    np.save(os.path.join('..','MRI_descattering','two_channel_result','SOS_gh.npy'),target.detach().numpy())


    
    

    

    

    

    

    
    

    
    

    
    
    
    
