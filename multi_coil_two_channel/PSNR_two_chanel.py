import torch
import os
import torch.nn as nn
import torch.nn.functional as F
#import dataset2
#import two_channel_dataset_test
import numpy as np
from Unet_model_fast_mri import Unet#,UNet
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
from math import log10, sqrt
from torch.utils.data.dataset import Dataset
from torch.nn import init
import two_channel_dataset_test
nChannels = 1
nClasses = 1
nEpochs = 30
#for i in range(len(kspace_array)): 
#    kspace_file = kspace_array[i]
#    kspace_data_from_file = np.load(os.path.join(kspace_data_name,kspace_file),'r')
#    kspace_data.append(kspace_data_from_file)


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


def make_vdrs_mask(N1, N2, nlines, init_lines):
    # Setting Variable Density Random Sampling (VDRS) mask
    mask_vdrs = np.zeros((N1, N2), dtype='bool')
    mask_vdrs[:, int(0.5 * (N2 - init_lines)):int(0.5 * (N2 + init_lines))] = True
    nlinesout = int(0.5 * (nlines - init_lines))
    rng = np.random.default_rng()
    t1 = rng.choice(int(0.5 * (N2 - init_lines)) - 1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(int(0.5 * (N2 + init_lines)) + 1, N2), size=nlinesout, replace=False)
    mask_vdrs[:, t1] = True;
    mask_vdrs[:, t2] = True
    return mask_vdrs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Unet(in_chans=2,
            out_chans=2)
# netG = UNet().to(device)
netG = netG.float()
norm = nn.L1Loss().to(device)
# tst=torch.zeros(nEpochs)
# vtst=torch.zeros(nEpochs)
# Loss and optimizer
#num_train =1
#test_noisy_paths = kspace_data[1000:1000+num_train]
#test_dataset = CustomDataset(test_clean_paths)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
for A_I_mask, A_I in two_channel_dataset_test.test_loader:
    A_I_mask = A_I_mask
    A_I = A_I
def PSNR2(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def NormalizeData2(data):
    return data / torch.max(torch.abs(data))
def convert_2chan_into_abs(img):
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    #show_img(torch.abs(img_complex).cpu(), cmap='gray')
    return img_complex

#A_I = np.load(os.path.join('..','MRI_descattering','two_channel_result','vali_gh_4_fold_ch1100_l2.npy'))
#A_I_mask = np.load(os.path.join('..','MRI_descattering','two_channel_result','vali_noise_4_fold_ch1100_l2.npy'))
netG = torch.load(os.path.join('..','MRI_descattering','Unet_global_model_data_two_channel_correct.pt'))
#A_I_local =np.load(os.path.join('..','MRI_descattering','two_channel_result','vali_image_4_fold_ch1100_l2.npy'))
with torch.no_grad():
    #vali_input = torch.from_numpy(A_I_mask).float()
    #vali_input = NormalizeData2(vali_input)
    vali_input = A_I_mask.float()
    #vali_input = NormalizeData2(vali_input)
    val_pred = netG(vali_input)
    #vali_local = torch.from_numpy(A_I_local).float()
    #vali_target = torch.from_numpy(A_I).float()
    vali_target = A_I.float()
    #vali_target = NormalizeData2(vali_target)
    #val_local = convert_2chan_into_abs(vali_local).numpy()
    val_pred = convert_2chan_into_abs(val_pred).numpy()
    vali_target = convert_2chan_into_abs(vali_target).numpy()
    vali_input =convert_2chan_into_abs(vali_input).numpy()
    print('input psnr', PSNR2(np.abs(vali_input), np.abs(vali_target)))
    print('global recons psnr', PSNR2(np.abs(val_pred), np.abs(vali_target)))


    
    
    
    
    
    #print('local recons psnr', PSNR2(np.abs(val_local), np.abs(vali_target)))

    

    

    
    
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_noise_global_0.1L1_ch1100.npy'),vali_input)  
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_image_global_0.1L1_ch1100.npy'),val_pred)    
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_gh_global_0.1L1_ch1100.npy'),vali_target)



    

    

    
    
    

    

    
    
