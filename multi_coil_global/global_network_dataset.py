import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import time
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import scipy.io as sio
#from ismrmrdtools import show, transform
# import ReadWrapper
from torch.utils.data.dataset import Dataset
from torch.nn import init
import math
import scipy
import scipy.linalg
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
from models import networks




#rt = '/home/liangs16/shared_data/mulitcoil_half'
Kspace_data_name = '/mnt/DataA/MRI_sampling/NEW_8_fold'
Kspace_data_name3 = '/mnt/DataA/NEW_KSPACE'
kspace_data = []
file = np.load(os.path.join('/mnt/DataA/MRI_sampling/file_list.npy'))
kspace_array = os.listdir(Kspace_data_name3)
kspace_array = sorted(kspace_array)
kspace_array2 = os.listdir(Kspace_data_name)
print(kspace_array2[1081])

clean_data1 = []
vali_data1 = []

##LOading the kspace data of size (sentive_map_real(coil,channel,size,size),sentive_map_img(coil,channel,size,size), kspace_real(coil,channel,size,size),kspace_img(coil,channel,size,size))

for i in range(1000): 
    kspace_file = file[i]
    kspace_data_from_file = np.load(os.path.join(Kspace_data_name,kspace_file),'r')
    if kspace_data_from_file['k_r'].shape[2]<373:
        clean_data1.append(kspace_data_from_file)
for j in range(1000): 
    kspace_file_other = kspace_array[j]
    kspace_data_from_file_other = np.load(os.path.join(Kspace_data_name3,kspace_file_other),'r')
    if kspace_data_from_file_other['k_r'].shape[2]<373 and kspace_data_from_file_other['k_r'].shape[2]>367:
        clean_data1.append(kspace_data_from_file_other)
for h in range(1074,1084):
    kspace_file_vali = file[h]
    kspace_vali_data_from_file = np.load(os.path.join(Kspace_data_name,kspace_file_vali),'r')
    if kspace_vali_data_from_file['k_r'].shape[2]<373:
        vali_data1.append(kspace_vali_data_from_file)

                      
mask_data_name = '/mnt/DataA/MRI_sampling/4_accerlation_mask'
mask_array = os.listdir(mask_data_name)
mask_array = sorted(mask_array)
mask_file = mask_array[1081]
mask_from_file = np.load(os.path.join(mask_data_name,mask_file),'r')
mask_data_selet = []
mask_vali = []
for e in range(len(clean_data1)):
    if clean_data1[e]['k_r'].shape[2]> 368:
        mask_data_selet.append(np.pad(mask_from_file, ((0, 0), (2, 2)), 'constant'))
    else:
        mask_data_selet.append(mask_from_file)
for k in range(len(vali_data1)):
    if vali_data1[k]['k_r'].shape[2]> 368:
        mask_vali.append(np.pad(mask_from_file, ((0, 0), (2, 2)), 'constant'))
    else:
        mask_vali.append(mask_from_file)


class nyumultidataset(Dataset): # model data loader
    def  __init__(self ,kspace_data,mask_data):
        self.A_paths = kspace_data
        #self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        #self.mask_path =sorted(self.mask_path)
        self.nx = 640
        self.ny = 368

    def __getitem__(self, index):
        A_temp = self.A_paths[index]
        s_r = A_temp['s_r']/ 32767.0 
        s_i = A_temp['s_i']/ 32767.0 
        k_r = A_temp['k_r']/ 32767.0
        k_i = A_temp['k_i']/ 32767.0 
        ncoil, nx, ny = s_r.shape
        mask = self.mask_path[index]
        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        mask = torch.tensor(np.repeat(mask[np.newaxis, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160], 2, axis=0), dtype=torch.float32)
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
        A_I = A_I/torch.max(torch.abs(SOS)[:])
        A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
        kreal = A_k
        AT = networks.OPAT2(A_s)
        Iunder = AT(kreal, mask)
        Ireal = AT(kreal, torch.ones_like(mask))
        return  Iunder, Ireal, A_s, mask
     
       
    def __len__(self):
        return len(self.A_paths)

len_data = len(clean_data1)


train_size = 0.9
num_train = 3000
    
train_clean_paths = clean_data1[:num_train]
mask_data_paths =mask_data_selet[:num_train]
train_dataset = nyumultidataset(train_clean_paths,mask_data_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True)
test_clean_paths = vali_data1 #1090
mask_test_paths = mask_vali
test_dataset = nyumultidataset(test_clean_paths,mask_test_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)





    

    

    
    
    
    
    
    
    
    
    
    

    


    
    

    

    
    
    
    
