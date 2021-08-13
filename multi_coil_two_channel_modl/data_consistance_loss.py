import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset2
import numpy as np
from Unet_model_fast_mri import Unet
import two_channel_dataset_test
import numpy as np
#from Unet_model_fast_mri import Unet
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
#from models import networks.CG as CG
from util.metrics import PSNR, roll_2
#import pytorch_msssim
from util.hfen import hfen 
import two_channel_dataset_data_consistance

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

nChannels = 1
nClasses = 1
nEpochs = 30


 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Unet(in_chans=2,
            out_chans=2).to(device)

netG = netG.float()
norm = nn.L1Loss().to(device)
# tst=torch.zeros(nEpochs)
# vtst=torch.zeros(nEpochs)
# Loss and optimizer
learning_rate = 1e-4
loss_G = nn.MSELoss()
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss = []
vali_loss = []

def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)

def NormalizeData2(data):
    return data / torch.max(torch.abs(data))
train_loss= []


for epoch in range(600):
    loss_G_train = 0

    for direct, target,smap,mask in two_channel_dataset_data_consistance.train_loader:
        input = direct.to(device).float()
        # print(input.shape)
        #CG = networks.CG.apply
        # input =torch.abs(input)
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        #print(mask.shape)
        label = target.to(device).float()
        #output = netG(input)
        # print(output.shape)
        for ii in range(6):
            output = netG(input)
            output = CG(output, tol =0.00005,L= 1, smap=smap, mask= mask, alised_image= input)
         

            
        #print(output.shape) 
       # print(result)
        # backward netG
        optimG.zero_grad()
        # if bfov: loss_G = fn(output * mask, label * mask)
        # loss_G = fn(output, label)+ 0.1*norm(output,label)
        loss_G = fn(output, label)
        loss_G.backward()
        optimG.step()
        loss_G_train += loss_G.item()  # torch.norm(label-output)/torch.norm(label)

    train_loss.append(loss_G_train)
    #vali_loss.append(val_loss.item())
    #print('V Loss', val_loss.item())
    print(loss_G_train)
    print(epoch)

    
    
        
        

        
        

        
        
        

        

