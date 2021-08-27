import torch
import torch.nn as nn
import torch.nn.functional as F
#import dataset2
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
nChannels = 1
nClasses=1
nEpochs =30


def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

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



#kspace_data_name = '/home/liangs16/MRI_descattering/RC_Image_4_fold_2channel/'
#kspace_data = []
#kspace_array = os.listdir(kspace_data_name)


#kspace_file = kspace_array[1181]
#kspace_data_from_file = np.load(os.path.join(kspace_data_name,kspace_file),'r')
#k_r = kspace_data_from_file['k_r'][()] # real channel part of image
#k_i = kspace_data_from_file['k_i'][()]

    
    
    #mask = A_temp['Q1'][()]
#nx, ny = k_r.shape
#k_r = np.reshape(k_r, [ nx, ny],order='A')
#k_i = np.reshape(k_i, [ nx, ny],order='A')
#k_np = np.stack((k_r, k_i), axis=0)
#mask = make_vdrs_mask(nx,ny,np.int(ny*0.14),np.int(ny*0.07))
#mask_two_channel = np.stack((mask,mask),axis=0)
#A_k = torch.tensor(k_np, dtype=torch.float32)
#A_I = ifft2(A_k.permute(1,2,0)).permute(2,0,1)
#A_k_mask =A_k
#A_k_mask[~mask_two_channel, :] = 0
#A_I_mask =ifft2(A_k_mask.permute(1,2,0)).permute(2,0,1)
#A_I = A_I[:, nx//2-nx//2:nx//2+nx//2,ny//2-ny//2:ny//2+ny//2]
#A_I_mask = A_I_mask[:, nx//2-nx//2:nx//2+nx//2,ny//2-ny//2:ny//2+ny//2]
#A_I_crop = center_crop(A_I, (320,320))
#A_I_mask_crop = center_crop(A_I_mask, (320,320))
    #A_I_mask = convert_2chan_into_abs(A_I_mask)


    





    

    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#netG = Unet(in_chans=2,
#            out_chans=2).to(device)
netG =Unet(2,2, num_pool_layers=2, chans=32).to(device)
#netG= UNet().to(device)
init_weights(netG, init_type='normal',init_gain=0.02)
norm =  nn.L1Loss().to(device)

# Loss and optimizer
learning_rate = 7e-5
loss_G = nn.MSELoss()
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss =[]
vali_loss =[]
def NormalizeData2(data):
    return data/torch.max(torch.abs(data))

for epoch in range(600):
    loss_G_train = 0

    for direct, target in two_channel_dataset_test.train_loader:
        input = direct.to(device).float()
        # print(input.shape)
        input = NormalizeData2(input)
         # input =torch.abs(input)
        label = target.to(device).float()
        label = NormalizeData2(label)
        output = netG(input)
        # backward netG
        optimG.zero_grad()
         #if bfov: loss_G = fn(output * mask, label * mask)
        loss_G = fn(output, label)+ 0.1*norm(output,label)

        loss_G.backward()
        optimG.step()
        loss_G_train += loss_G.item()#torch.norm(label-output)/torch.norm(label)
    with torch.no_grad():
         for vali_direct,vali_target in two_channel_dataset_test.test_loader:
            vali_input =  vali_direct.to(device).float()
            val_pred = netG(vali_input)
            vali_target = vali_target.to(device).float()    
            val_loss = fn(val_pred, vali_target)

       # vali_input = A_I_mask_crop.unsqueeze(0).to(device).float()
       # val_pred = netG(vali_input2)
       # vali_target = A_I_crop.unsqueeze(0).to(device).float()
       # val_loss = fn(val_pred, vali_target2)
    

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


















    
    
    
    
