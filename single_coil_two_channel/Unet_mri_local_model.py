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

nChannels = 1
nClasses=1
nEpochs =30

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]









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




#clean_data = []
#clean_data_name = '/home/liangs16/MRI_descattering/GT_image_4_fold/'
#noisy_data_name = '/home/liangs16/MRI_descattering/RC_image_4_fold/'
#noisy_data = []
#noisy_array = os.listdir(noisy_data_name)
#clean_array = os.listdir(clean_data_name)
#noisy_array.sort()

#clean_array.sort()
#print(clean_array)
#for i in range(11,12): #13,14 11,12
#    noisy_file = noisy_array[i]
#    clean_file = clean_array[i]
#    clean_data_from_file = np.load(os.path.join(clean_data_name,clean_file),'r')
#    noisy_data_from_file = np.load(os.path.join(noisy_data_name,noisy_file),'r')
#    noisy_data.append(noisy_data_from_file['arr_0'][25])
#    clean_data.append(clean_data_from_file['arr_0'][25])
   # noisy_data.append(np.load(os.path.join('..','MRI_descattering','vali_image_early_datasset_0.1L1_20noisy.npy'))[0][0])
kspace_data_name = '/home/liangs16/MRI_descattering/RC_Image_4_fold_2channel/'
kspace_data = []
kspace_array = os.listdir(kspace_data_name)
print(len(kspace_array))


kspace_file = kspace_array[1180]
kspace_data_from_file = np.load(os.path.join(kspace_data_name,kspace_file),'r')
k_r = kspace_data_from_file['k_r'][()] # real channel part of image
k_i = kspace_data_from_file['k_i'][()]

    
    
    #mask = A_temp['Q1'][()]
nx, ny = k_r.shape
k_r = np.reshape(k_r, [ nx, ny],order='A')
k_i = np.reshape(k_i, [ nx, ny],order='A')
k_np = np.stack((k_r, k_i), axis=0)
mask = make_vdrs_mask(nx,ny,np.int(ny*0.14),np.int(ny*0.06))
mask_two_channel = np.stack((mask,mask),axis=0)
A_k = torch.tensor(k_np, dtype=torch.float32)
A_I = ifft2(A_k.permute(1,2,0)).permute(2,0,1)
A_k_mask =A_k
A_k_mask[~mask_two_channel, :] = 0
A_I_mask =ifft2(A_k_mask.permute(1,2,0)).permute(2,0,1)
A_I = A_I[:, nx//2-nx//2:nx//2+nx//2,ny//2-ny//2:ny//2+ny//2]
A_I_mask = A_I_mask[:, nx//2-nx//2:nx//2+nx//2,ny//2-ny//2:ny//2+ny//2]
A_I = center_crop(A_I, (320,320))
A_I_mask = center_crop(A_I_mask, (320,320))
    #A_I_mask = convert_2chan_into_abs(A_I_mask)


    



    


    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#netG = Unet(in_chans=2,
#            out_chans=2).to(device)
netG =Unet(2,2, num_pool_layers=3, chans=32).to(device)
#netG= UNet().to(device)
netG = netG.float()
norm =  nn.L1Loss().to(device)
#tst=torch.zeros(nEpochs)
#vtst=torch.zeros(nEpochs)
# Loss and optimizer
learning_rate = 8e-5
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
        # print(output.shape)    

         # backward netG
        optimG.zero_grad()
         #if bfov: loss_G = fn(output * mask, label * mask)
        loss_G = fn(output, label)+ 0.1*norm(output,label)

        loss_G.backward()
        optimG.step()
        loss_G_train += loss_G.item()#torch.norm(label-output)/torch.norm(label)
    with torch.no_grad():
        #vali_input = np.abs(noisy_data[0])
        #vali_input = torch.from_numpy(vali_input).unsqueeze(0).unsqueeze(0).to(device).float()
        vali_input = A_I_mask.unsqueeze(0).to(device).float()
        vali_input = NormalizeData2(vali_input)
        val_pred = netG(vali_input)
        #vali_target = np.abs(clean_data[0])
        vali_target = A_I.unsqueeze(0).to(device).float()
    #    vali_target = torch.from_numpy(vali_target).unsqueeze(0).unsqueeze(0).to(device).float()
        vali_target = NormalizeData2(vali_target)
        val_loss = fn(val_pred, vali_target)
        # for j in range(len(noisy_data)):
              #vali_input = np.abs(noisy_data[0])
              #vali_input = torch.from_numpy(vali_input).unsqueeze(0).unsqueeze(0).to(device).float()
              #vali_input = NormalizeData2(vali_input)
              #val_pred = netG(vali_input)
              #vali_target = np.abs(clean_data[0])
              #vali_target = torch.from_numpy(vali_target).unsqueeze(0).unsqueeze(0).to(device).float()
              #vali_target = NormalizeData2(vali_target)
              #val_loss = fn(val_pred, vali_target)


    #train_loss.append(loss_G_train)
   # vali_loss.append(val_loss.item())
    print('V Loss', val_loss.item())
    print(loss_G_train)
    print(epoch)

    
    
    

    
    
    

    
    

    #with torch.no_grad():                                    
#    vali_input = np.abs(noisy_data[0])                                                                          
#    vali_input = torch.from_numpy(vali_input).unsqueeze(0).unsqueeze(0).to(device).float()                      
#    vali_input = NormalizeData2(vali_input)                                                                     
#    val_pred = netG(vali_input)                                                                                
#    vali_target = np.abs(clean_data[0])                                                                         
#    vali_target = torch.from_numpy(vali_target).unsqueeze(0).unsqueeze(0).to(device).float()                   
#    vali_target = NormalizeData2(vali_target)                                                                   
#    val_loss = fn(val_pred, vali_target)  
    
    
    
    
    

#torch.save(netG.cpu(),'Unet_two_channel_model_1400_image.pt')








    
    
    
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_gh_4_fold_ch1100_l2.npy'),vali_target.cpu().numpy())

np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_image_4_fold_ch1100_l2.npy'),val_pred.cpu().numpy())
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_noise_4_fold_ch1100_l2.npy'),vali_input.cpu().numpy())
#np.save(os.path.join('..','MRI_descattering','two_channel_result','train_noise_1400_datasset_0.1L1_two_ch1100.npy'),input.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','two_channel_result','train_image_1400_datasset_0.1L1_two_ch1100.npy'),output.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','two_channel_result','train_gh_1400_datasset_0.1L1__two_ch1100.npy'),label.cpu().detach().numpy())    
np.save(os.path.join('..','MRI_descattering','two_channel_result','val_gh_4_fold_loss_cg1100.npy'),vali_loss)


np.save(os.path.join('..','MRI_descattering','two_channel_result','train_gh_1400_0.1L1__two_ch1100.npy'),train_loss) 















    
    
    
    
