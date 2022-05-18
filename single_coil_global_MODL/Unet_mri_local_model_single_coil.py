import torch
import torch.nn as nn
import torch.nn.functional as F
import two_channel_dataset_test
import numpy as np
from Unet_model_fast_mri import Unet,UNet
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2,ifft2_mask, cplx_to_tensor, complex_conj, complex_matmul, absolute
from torch.nn import init
from models import networks
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG =Unet(2,2, num_pool_layers=4, chans=64).to(device)
init_weights(netG, init_type='normal',init_gain=0.02)
norm =  nn.L1Loss().to(device)
# Loss and optimizer
learning_rate = 4e-5
loss_G = nn.MSELoss()
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss =[]
vali_loss =[]

for epoch in range(600):
    loss_G_train = 0
    vali_g_train = 0
    for direct, target,mask in two_channel_dataset_test.train_loader:
        input = direct.to(device).float()
        #input = NormalizeData2(input)
        label = target.to(device).float()
        #label = NormalizeData2(label)
        mask = mask.to(device).float()
        smap = torch.ones(label.shape).unsequeeze()
        temp = input
        for ii in range(6):
            output = netG(temp)
            output2 = CG(output, tol=0.00005, L=1, smap=smap, mask=mask, alised_image=input)
        temp = output2
        optimG.zero_grad()
         #if bfov: loss_G = fn(output * mask, label * mask)
        loss_G = fn(temp, label)
        loss_G.backward()
        optimG.step()
        loss_G_train += loss_G.item()#torch.norm(label-output)/torch.norm(label)
    with torch.no_grad():
         for vali_direct,vali_target,vali_mask in two_channel_dataset_test.test_loader:
            vali_input =  vali_direct.to(device).float()
            val_pred = netG(vali_input)
            vali_temp = vali_input
            vali_mask = vali_mask.to(device).float()
            vali_smap = torch.ones(vali_input.shape).unsequeez()
            for ii in range(6):
                vali_output = netG(vali_input)
                vali_output2 = CG(vali_output, tol=0.00005, L=1, smap=vali_smap, mask=vali_mask, alised_image=vali_input)
            vali_temp = vali_output2
            vali_target = vali_target.to(device).float()    
            val_loss = fn(vali_temp, vali_target)
         vali_g_train += val_loss.item() 
    train_loss.append(loss_G_train)
    vali_loss.append(val_loss.item())
    print('V Loss', val_loss.item())
    print(loss_G_train)
    print(epoch)


    

    
    
    
    

    
    

    
    
    

#torch.save(netG.cpu(),'Unet_two_channel_model_1400_image_8accelreat_corr3.pt')


    


np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_gh_4_fold_ch1180_new_norm2.npy'),vali_target.cpu().numpy())

np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_image_4_fold_ch1180_new_norm2.npy'),val_pred.cpu().numpy())
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_noise_4_fold_ch1180_new_norm2.npy'),vali_input.cpu().numpy())



#np.save(os.path.join('..','MRI_descattering','two_channel_result','train_noise_1400_datasset_0.1L1_two_ch1100.npy'),input.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','two_channel_result','train_image_1400_datasset_0.1L1_two_ch1100.npy'),output.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','two_channel_result','train_gh_1400_datasset_0.1L1__two_ch1100.npy'),label.cpu().detach().numpy())    
np.save(os.path.join('..','MRI_descattering','two_channel_result','val_gh_8_fold_loss_cg1180_loss_4acc.npy'),vali_loss)


np.save(os.path.join('..','MRI_descattering','two_channel_result','train_gh_1180_two_2chloss_8acc.npy'),train_loss) 


















    
    
    
    
