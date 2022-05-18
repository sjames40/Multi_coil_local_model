import numpy as np
#import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import time
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.nn import init
import math
import scipy
import scipy.linalg
#import read_ocmr as read
import h5py
import sys
import os
import torch
#from MRI_sampling import data
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
from models import networks
from util.util import convert_2chan_into_complex,make_data_list


Kspace_data_name= '/mnt/DataA/NEW_KSPACE'
kspace_array = os.listdir(Kspace_data_name)
kspace_array = sorted(kspace_array)
kspace_data = []

# loading the 3000 image inside the kspace data which is the folder of the kspace image
for j in range(len(kspace_data): 
    kspace_file = kspace_array[j]
    kspace_data_from_file = np.load(os.path.join(Kspace_data_name,kspace_file),'r')
    if kspace_data_from_file['k_r'].shape[2]<373 and kspace_data_from_file['k_r'].shape[2]>367:
        kspace_data.append(kspace_data_from_file)

# loading the 4 accleration mask and all the training  undersample  space image to facilltate the neighbor searching    

image_data_name ='/home/shijunliang/Newproject/four_fold_image_shape'
image_space_array = os.listdir(image_data_name)
image_space_array = sorted(image_space_array)
# loading the test undersample image space to avoid the confusion between the training noisy image 
image_test_name = '/home/shijunliang/Newproject/test_four_fold'
image_test_array = os.listdir(image_test_name)
image_test_array = sorted(image_test_array)
mask_data_name = '/mnt/DataA/4acceleration_mask_random3'
mask_array = os.listdir(mask_data_name)
mask_array = sorted(mask_array)
mask_data_name_test = '/mnt/DataA/4acceleration_mask_test2'
mask_array_test = os.listdir(mask_data_name_test)
mask_array_test = sorted(mask_array_test)

# make the data list for training noisy image , test noisy image and the specific mask 1081 
image_test_data = make_data_list(image_test_name,image_test_array)# test four fold
image_test_gt_data = make_data_list(image_test_gt_name,image_test_gt_array)# test gt image
mask_data_set_train = make_data_list(mask_data_name,mask_array)
mask_data_set_test = make_data_list(mask_data_name_test,mask_array_test)# load mask

# compute the neighbor between the reference image which is the 1074 at the beigning of the test data 

def search_for_simliar_neighbor(test_image,training_dataset,metric,number_neighbor):
    #disatnce =0
    norm_matrix2 = []
    for sample in range(len(training_dataset)):
        neighbor = training_dataset[sample]
        if metric == 'L1':
            distance = np.mean(np.abs(np.abs(test_image)-np.abs(neighbor)),axis=(0,1))  
        elif metric == 'L2':
            distance =np.linalg.norm(np.abs(test_image)-np.abs(neighbor),'fro')
        elif metric == 'cos':
            distance = np.abs(np.sum(np.conj(test_image)*neighbor))/(np.linalg.norm(test_image)*np.linalg.norm(neighbor))
        else:
            distance = 0
        norm_matrix2.append(distance)
    if metric == 'cos':
       match_inds = np.argsort(norm_matrix2)[-number_neighbor:-1]
    if metric == 'L1':
       match_inds = np.argsort(norm_matrix2)[1:number_neighbor+1]
    if metric == 'L2':
       match_inds = np.argsort(norm_matrix2)[1:number_neighbor+1]
    check_inds = sorted(match_inds)
    print(check_inds)
    return check_inds
def make_dataset_with_output(output,index,number_neigh,train_data_path,metric):
    data_select = []
    mask_data_select =[]
    match_inds = search_for_simliar_neighbor(output,train_data_path,metric,number_neigh)
    for b in range(len(match_inds)):
        kspace_image_data_select= kspace_data[(match_inds[b])]
        data_select.append(kspace_image_data_select)
        mask_data_set_train_data_select = mask_data_set_train[(match_inds[b])]
        mask_data_select.append(mask_data_set_train_data_select)                
    return data_select,mask_data_select

data_select,mask_data_select = make_dataset_with_output(image_test_data[number],1078,number_of_neighbor,image_data_name,metric)
number = 3
number_of_neighbor =10

vali_data1 = []
mask_vali = []


index = 1078                
kspace_file_vali = file_list[index]
kspace_vali_data_from_file = np.load(os.path.join(Kspace_data_name_1400,kspace_file_vali),'r')
vali_data1.append(kspace_vali_data_from_file)
mask_data_vali_select = mask_data_set_test[number]
mask_vali.append(mask_data_vali_select)



class nyumultidataset(Dataset): # model data loader
    def  __init__(self ,kspace_data,mask_data):
        self.A_paths = kspace_data
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.mask_path = mask_data
        self.mask_path =sorted(self.mask_path)
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
        ##A_s is the sensitive map 
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



train_dataset = nyumultidataset(data_select,mask_data_select)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True)
test_clean_paths = vali_data1 #1090
mask_test_paths = mask_vali
test_dataset = nyumultidataset(test_clean_paths,mask_test_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False)




    

    

    
    

    

    
    
    
    
