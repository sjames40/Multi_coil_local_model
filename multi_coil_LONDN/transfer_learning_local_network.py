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
#from models.base_model import BaseModel
from models import networks
from util.metrics import PSNR, roll_2
from util.hfen import hfen 
#import two_channel_dataset_8accerlation
import local_network_dataset
from torch.nn import init
from models.didn import DIDN
from util.util import convert_2chan_into_complex
from util.metric import PSNR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = DIDN(2, 2, num_chans=32,pad_data=True, global_residual=True, n_res_blocks=2)

init_weights(netG, init_type='normal',init_gain=0.02)

netG = torch.load(os.path.join('..','Newproject','DIDN_global_model_142iteration_3res_6iter.pt'))

netG = netG.float().to(device)
norm = nn.L1Loss().to(device)
# Loss and optimizer
learning_rate = 8e-5
epoch = 30
cg_iter = 6
regu_parameter =0.000000001
train_number =30
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss = []
vali_loss = []


def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)

PSNR_list =[]
train_loss= []
vali_loss =[]


for epoch in range(epoch):
    loss_G_train = 0
    
    for direct, target,smap,mask in two_channel_dataset_faster_load.train_loader:    
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
           l1_regularization += torch.norm(param,1)
        loss_G = loss_G + regu_parameter*l1_regularization 
        loss_G.backward()
        optimG.step()
        loss_G_train += float(loss_G)  
    with torch.no_grad():
        for vali_direct, vali_target,vali_smap,vali_mask in  two_channel_dataset_faster_load.test_loader:
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
        local_gt = vali_label
        local_rec = vali_result
        local_gt2 =convert_2chan_into_complex(local_gt).cpu().detach().numpy()
        local_rec2 =convert_2chan_into_complex(local_rec).cpu().detach().numpy() 
        PSNR_rec = PSNR(np.abs(local_gt2),np.abs(local_rec2))
    train_loss.append(loss_G_train/train_number)
    vali_loss.append(val_loss.item())
    PSNR_list.append(PSNR_rec)
    print('V Loss', val_loss.item())
    print(loss_G_train/train_number)
    print(epoch)
    print(PSNR_rec)

    
    


