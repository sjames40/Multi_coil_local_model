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

rt = '/mnt/shared_b/data/fastMRI/singlecoil_train/'
aim = '/home/liangs16/labmat_project/MRI_descattering/NEW_8_fold'
os.chdir(rt)
foldername = []
for i in os.listdir():
    if i.endswith('h5'):
        foldername.append(i)
foldername.sort()

arr = os.listdir(rt)
count =0
for i in range(30,40):#Lines: #len(arr)#15.20
    print(i)
    count=count+1
    filename1 = arr[i]
    hf = h5py.File(os.path.join(rt,filename1), 'r')
    kspace = hf['kspace'][()]
    #print(kspace)
    num_layers, nx, ny = kspace.shape
    kmax = np.amax(np.abs(kspace[:]))
    for j in range(num_layers):
        k = kspace[j,:,:]
        k_r = (np.real(k)/kmax*32767).astype('int16') # save memory, should be consistent with your dataloader!
        k_i = (np.imag(k)/kmax*32767).astype('int16')
        #print(k_i.dtype)
        #print(k_i/32767)

        np.savez(os.path.join( '/home/liangs16/MRI_descattering/RC_Image_4_fold_2channel','%s_Layer%d.npz'%(filename1,j)),k_r=k_r,k_i=k_i)

        
        
        
        
