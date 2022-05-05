import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
from util.image_pool import ImagePool
from models import networks
from util.metrics import PSNR, roll_2
from util.hfen import hfen 
import local_network_dataset
from Unet_model_fast_mri_shallow import Unet
from util.util import convert_2chan_into_complex, init_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Unet(input_channel=2,nclass = 2, num_pool_layers=2,chans=64).to(device)
init_weights(netG, init_type='normal',init_gain=0.01)
netG = netG.float()
learning_rate = 8e-5 
epoch = 600
regu_parameter =0.0000000001
cg_iter = 6
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss = []
vali_loss = []


def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)

PSNR_list =[]
train_loss= []
vali_loss =[]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=[100,200], gamma=0.5)

for epoch in range(epoch):
    loss_G_train = 0
    
    for direct, target,smap,mask in two_channel_dataset_4acc_random_mask.train_loader:    
        noise_input = direct.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        label = target.to(device).float()
        temp = noise_input
        for ii in range(cg_iter):
            output = netG(temp)
            output2 = CG(output, tol =0.00001,L= 0.1, smap=smap, mask= mask, alised_image= noise_input).type(torch.float32)
            temp = output2
        output_final = temp
        optimG.zero_grad()
        loss_G = fn(output_final, label)
        l1_regularization =torch.tensor(0).to(device).float()
        for param in netG.parameters():
           l1_regularization += torch.norm(param, 1)
        loss_G = loss_G + regu_parameter*l1_regularization
        loss_G.backward()
        optimG.step()

        loss_G_train += float(loss_G) 
    with torch.no_grad():
        for vali_direct, vali_target,vali_smap,vali_mask in two_channel_dataset_4acc_random_mask.test_loader:
            vali_input = vali_direct.to(device).float()
            vali_smap = vali_smap.to(device).float()
            vali_mask = vali_mask.to(device).float()
            vali_label = vali_target.to(device).float()
            vali_temp = vali_input
            for jj in range(cg_iter):
                vali_output = netG(vali_temp)
                vali_output2 = CG(vali_output, tol =0.00001,L= 0.1, smap=vali_smap, mask= vali_mask, alised_image= vali_input).type(torch.float32)
                vali_temp = vali_output2
            vali_result = vali_temp
            val_loss = fn(vali_result,vali_label)+regu_parameter*l1_regularization            
            val_loss = val_loss+l1_regluarization
        local_gt = vali_label
        local_rec = vali_result
        local_gt2 =convert_2chan_into_abs(local_gt).cpu().detach().numpy()
        local_rec2 =convert_2chan_into_abs(local_rec).cpu().detach().numpy() 
        PSNR_rec = PSNR(np.abs(local_gt2),np.abs(local_rec2))
    scheduler.step()
    train_loss.append(loss_G_train/35)
    vali_loss.append(val_loss.item())
    PSNR_list.append(PSNR_rec)
    print('V Loss', val_loss.item())
    print(loss_G_train/35)
    print(epoch)
    print(PSNR_rec)

    
    
    
















