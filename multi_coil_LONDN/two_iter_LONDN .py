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
from util.util import convert_2chan_into_complex, init_weights



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG =Unet(2,2, num_pool_layers=2,chans=64)# residue block 1 or2 with channel 128

init_weights(netG, init_type='normal',init_gain=0.02)

norm = nn.L1Loss().to(device)
# Loss and optimizer
learning_rate =  4e-5
#loss_G = nn.MSELoss()
fn = nn.MSELoss().to(device)
train_loss = []
vali_loss = []



def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)
def PSNR(img1, img2):
    MSE = np.mean(np.abs(img1-img2)**2)
    psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #psnr = 10*math.log10(float(1.**2)/MSE)
    return psnr

PSNR_list =[]
train_loss= []
vali_loss =[]

train_loader_initial = tp.train_loader
bileveloptimzation_loss_list =[]
for number in range(2):
    for epoch in range():
        loss_G_train = 0
        for direct, target,smap,mask in train_loader_initial:
            noise_input = direct.to(device).float()
            smap = smap.to(device).float()
            mask = mask.to(device).float()
            label = target.to(device).float()
            temp = noise_input
            for ii in range(5):
                output = netG(temp)
                output2 = CG(output, tol =0.00001,L= 0.1, smap=smap, mask= mask, alised_image= noise_input).type(torch.float32)
                temp = output2
            output_final = temp
            optimG.zero_grad()
            loss_G = fn(output_final, label)
            l1_regularization =torch.tensor(0).to(device).float()
            for param in netG.parameters():
                l1_regularization += torch.norm(param, 1)#**2
            loss_G = loss_G + 0.00000001*l1_regularization #0.0000001
            loss_G.backward()
            optimG.step()
            loss_G_train += float(loss_G)
        with torch.no_grad():
            for vali_direct, vali_target,vali_smap,vali_mask in  tp.test_loader:
                vali_input = vali_direct.to(device).float()
                vali_smap = vali_smap.to(device).float()
                vali_mask = vali_mask.to(device).float()
                vali_label = vali_target.to(device).float()
                vali_temp = vali_input
                for jj in range(5):
                    vali_output = netG(vali_temp)
                    vali_output2 = CG(vali_output, tol =0.00001,L= 0.1, smap=vali_smap, mask= vali_mask, alised_image= vali_input).type(torch.float32)
                    vali_temp = vali_output2
                vali_result = vali_temp
                val_loss = fn(vali_result,vali_label)+ 0.00000001*l1_regularization 
            local_gt = vali_label
            local_rec = vali_result
            local_gt2 =convert_2chan_into_abs(local_gt).cpu().detach().numpy()
            local_rec2 =convert_2chan_into_abs(local_rec).cpu().detach().numpy() 
            PSNR_rec = PSNR(np.abs(local_gt2),np.abs(local_rec2))
        print(PSNR_rec)
    check_image_set =[]
    data_select,mask_data_select,check_image_set = tp.make_dataset_with_output(local_rec2,1074,tp.image_gt_space_data,tp.kspace_data,25,'Cos')
    train_dataset = tp.nyumultidataset(data_select,mask_data_select)
    train_loader_new = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True)
    train_loader_initial = train_loader_new


                                           
     

    
    
    
    














