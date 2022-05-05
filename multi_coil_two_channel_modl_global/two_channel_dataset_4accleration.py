import numpy as np
import matplotlib.pyplot as plt
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
import glob


kspace_data_name = '/home/shijunliang/MRI_sampling/NEW_8_fold'
kspace_data_file = glob.glob(os.path.join('..','MRI_sampling','NEW_8_fold','*.npz'))
#loading the kspace data

kspace_data = []

# loading the mask
mask_data_name = '/home/shijunliang/MRI_sampling/4_accerlation_mask'
mask_array = os.listdir(mask_data_name)
mask_data =[]
mask_array = sorted(mask_array)
for c in range(len(mask_array)):
    mask_file = mask_array[c]
    mask_from_file = np.load(os.path.join(mask_data_name,mask_file),'r')
    mask_data.append(mask_from_file)


image_data_name ='/home/shijunliang/MRI_sampling/clean_image_8_fold'
image_space_array = os.listdir(image_data_name)
image_space_array = sorted(image_space_array)

# loading the image space data for neighbor search and save time
image_space_data =[]
for i in range(len(image_space_array)): 
    image_space_file = image_space_array[i]
    image_space_data_from_file = np.load(os.path.join(image_data_name,image_space_file),'r')
    image_space_data.append(image_space_data_from_file)


def convert_2chan_into_abs(img):
    img_real = img[0]
    img_imag = img[1]
    img_complex = torch.complex(img_real, img_imag)
    return img_complex
def convert_2chan_into_abs_2(img):
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    return img_complex



number = 1083
norm_matrix =[]
norm_matrix2 =[]
number_of_neighbor = 30
for a in range(len(image_space_array)):
    norms =np.linalg.norm(np.abs(image_space_data[number])-np.abs(image_space_data[a]),'fro')
    norm_matrix.append(norms)
    match_inds = np.argsort(norm_matrix)[1:number_of_neighbor +1]

print(match_inds)

                           
clean_data1 = []
mask_data_selet =[]
number =1083
# making the neighbor dataset
for b in range(len(match_inds)):
    image_data_select= np.load(kspace_data_file[(match_inds[b])])
    if image_data_select['k_r'].shape[2]<373:
        clean_data1.append(image_data_select)
reference = np.load(kspace_data_file[number])
print(reference['k_r'].shape)

print(mask_data[1081].shape)
# loading the image with same mask

if reference['k_r'].shape[2]<372:
    for e in range(len(clean_data1)):
        if clean_data1[e]['k_r'].shape[2]> 368:
          # mask_data_selet.append(mask_data[number][:,2:372-2])
           mask_data_selet.append(np.pad(mask_data[number], ((0, 0), (2, 2)), 'constant'))
        else:
           mask_data_selet.append(mask_data[number])
if reference['k_r'].shape[2]>368:
    for f in range(len(clean_data1)):
        if clean_data1[f]['k_r'].shape[2]<=368:
           mask_data_selet.append(mask_data[number][:,2:372-2]) 
           #mask_data_selet.append(np.pad(mask_data[number], ((0, 0), (2, 2)), 'constant'))
        else:
           mask_data_selet.append(mask_data[number])


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
        # loading the image real and imagenary
        # loading the sentivity maps with real and imagenary
        mask = self.mask_path[index]
        ncoil, nx, ny = s_r.shape
        k_np = np.stack((k_r, k_i), axis=0)
        s_np = np.stack((s_r[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160],
                         s_i[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]), axis=0)
        # cropping the image
        mask = np.stack((mask,mask),axis=0)
        mask =torch.tensor(mask)
        mask_two_channel = mask.repeat(ncoil, 1, 1, 1)
        #loading the mask and make the image to same shape
        A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3)
        mask_two_channel =mask_two_channel#.numpy()
        ##A_I is the ifft of the kspace 
        A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A_I = A_I[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        ##A_s is the sensitive map 
        A_s = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)), dim=0)
        # reconstruct the mulit coil image with senstivity maps
        A_I_crop = SOS / torch.max(torch.abs(SOS))
        # normlize the image
       
        A_k_mask = A_k
        A_k_mask = A_k_mask*mask_two_channel
        A_I_mask =ifft2(A_k_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 
        A_I_mask = A_I_mask[:, :, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        A_s_mask = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3)
        SOS_mask = torch.sum(complex_matmul(A_I_mask, complex_conj(A_s_mask)), dim=0)
        # print('sossize',SOS.size())
        A_I_mask_crop = SOS_mask / torch.max(torch.abs(SOS_mask))
        mask = mask[:, nx // 2 - 160:nx // 2 + 160, ny // 2 - 160:ny // 2 + 160]
        
        return  A_I_mask_crop, A_I_crop, A_s, mask
     
       
    def __len__(self):
        return len(self.A_paths)

    
len_data = len(image_space_data)

test_path =[]
test_path.append(np.load(kspace_data_file[number]))
#test_mask1 = []
#test_mask1.append(np.pad(mask_data[1081], ((0, 0), (2, 2)), 'constant'))
train_dataset = nyumultidataset(clean_data1,mask_data_selet)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,shuffle=True)
#test_clean_paths = kspace_data[1083:1083+1]
test_clean_paths = test_path#np.load((kspace_data_file[1083])
test_mask = mask_data[number:number+1]
test_mask =test_mask

test_dataset = nyumultidataset(test_clean_paths,test_mask)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)


    
    
    

    


    
    

    

    
    
    
    
