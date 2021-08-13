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
import sys

#path = os.environ["/home/liangs16/bart-0.7.00"] + "/python/";
#sys.path.append(path);

from bart import bart



# import ReadWrapper
import read_ocmr as read
import h5py
import sys
import os
import torch

rt = '/home/liangs16/shared_data/mulitcoil_half'
aim = '/home/liangs16/labmat_project/MRI_descattering/NEW_8_fold'
os.chdir(rt)
foldername = []
for i in os.listdir():
    if i.endswith('h5'):
        foldername.append(i)
foldername.sort()


def smap(filename,filename1):
    aim = '/home/liangs16/labmat_project/MRI_descattering/NEW_8_fold'
    hf = h5py.File(os.path.join(filename,filename1), 'r')
    kspace = hf['kspace'][()]
    num_layers, num_coils, nx, ny = kspace.shape
    kmax = np.amax(np.abs(kspace[:]))
    for i in range(num_layers):
        k = kspace[i,:,:,:]
        k_r = (np.real(k)/kmax*32767).astype('int16') # save memory, should be consistent with your dataloader!
        k_i = (np.imag(k)/kmax*32767).astype('int16')
        
        k = np.expand_dims(k, axis=0)
        k = np.transpose(k,(2,3,0,1))
        try:
            sens_maps = bart(1, f'ecalib -d0 -m1 -r24 -k6 -t0.001 -c0.95', k)
            print(sens_maps)
        except:
            print('oops')
            continue
        sens_map = sens_maps.squeeze().transpose((2,0,1))
        s_r = (np.real(sens_map)*32767).astype('int16')
        s_i = (np.imag(sens_map)*32767).astype('int16')
        np.savez(os.path.join(aim,'%s_Layer%d.npz'%(filename,i)),s_r=s_r,s_i=s_i,k_r=k_r,k_i=k_i)

arr = os.listdir(rt)
print(len(arr))

count =0

for i in range(1,len(arr)):#Lines: #len(arr)
    print(i)
    count=count+1
    filename1 = arr[i]
    hf = h5py.File(os.path.join(rt,filename1), 'r')
    filename1 = arr[i]
    kspace = hf['kspace'][()]
    num_layers, num_coils, nx, ny = kspace.shape
    kmax = np.amax(np.abs(kspace[:]))
    for j in range(num_layers):
        k = kspace[j,:,:,:]
        k_r = (np.real(k)/kmax*32767).astype('int16') # save memory, should be consistent with your dataloader!
        k_i = (np.imag(k)/kmax*32767).astype('int16')
        k = np.expand_dims(k, axis=0)
        k = np.transpose(k,(2,3,0,1))
        try:
            sens_maps = bart(1, f'ecalib -d0 -m1 -r24 -k6 -t0.001 -c0.95', k)
        except:
            print('oops')
            continue
        sens_map = sens_maps.squeeze().transpose((2,0,1))
        s_r = (np.real(sens_map)*32767).astype('int16')
        s_i = (np.imag(sens_map)*32767).astype('int16')
        np.savez(os.path.join('/home/liangs16/labmat_project/MRI_descattering/NEW_8_fold','%s_Layer%d.npz'%(filename1,j)),s_r=s_r,s_i=s_i,k_r=k_r,k_i=k_i)


        
        
        
        
        

        
