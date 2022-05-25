import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Unet_model_fast_mri import Unet
#import two_channel_dataset_test
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
from util.image_pool import ImagePool
#from models.base_model import BaseModel
from models import networks
from models.networks import UnetGenerator
from util.metrics import PSNR, roll_2
#import pytorch_msssim
from util.hfen import hfen
import global_network_dataset
from util.util import convert_2chan_into_complex, init_weights
#from models.didn import DIDN
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Unet(2,2, num_pool_layers=4,chans=64).to(device)



init_weights(netG, init_type='normal',init_gain=0.02)
netG = netG.float().to(device)
norm = nn.L1Loss().to(device)
# Loss and optimizer
learning_rate = 1e-4
epoch = 150
cg_iter =6
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss = []
vali_loss = []
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=[50,100], gamma=0.6)

def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)

PSNR_list =[]
train_loss= []
vali_loss =[]



for epoch in range(epoch):
    loss_G_train = 0
    for direct, target,smap,mask in global_network_dataset.train_loader:    
        input = direct.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        label = target.to(device).float()
        temp = input
        for ii in range(cg_iter):
            output = netG(temp)
            output2 = CG(output, tol =0.00001,L= 0.1, smap=smap, mask= mask, alised_image= input)
            temp = output2
        output_final = temp
        optimG.zero_grad()
        loss_G = fn(output_final, label)
        loss_G.backward()
        optimG.step()
        loss_G_train += float(loss_G) 
    with torch.no_grad():
        vali_loss_total = 0
        for vali_direct, vali_target,vali_smap,vali_mask in global_network_dataset.test_loader:
            vali_input = vali_direct.to(device).float()
            vali_smap = vali_smap.to(device).float()
            vali_mask = vali_mask.to(device).float()
            vali_label = vali_target.to(device).float()
            vali_temp = vali_input
            for jj in range(cg_iter):
                vali_output = netG(vali_temp)
                vali_output2 = CG(vali_output, tol =0.00001,L= 0.1, smap=vali_smap, mask= vali_mask, alised_image= vali_input)
                vali_temp = vali_output2
            vali_result = vali_temp
            Loss_vail = fn(vali_result,vali_label)
            vali_loss_total += float(Loss_vail)
    train_loss.append(loss_G_train/2000)
    vali_loss.append(vali_loss_total/5)
    print('V Loss',vali_loss_total/5)
    print(loss_G_train/2800)
    print(epoch)










