import torch
import torch.nn as nn
import torch.nn.functional as F
#import dataset2
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
#import two_channel_dataset_8accelration
import two_channel_dataset_4accleration
from torch.nn import init



## function to convert two channel image to image with complex and real value
def convert_2chan_into_abs(img):
    img_real = img[0][0]
    img_imag = img[0][1]
    img_complex = torch.complex(img_real, img_imag)
    #show_img(torch.abs(img_complex).cpu(), cmap='gray')
    return img_complex


# initilize the network
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


    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Unet(2,2, num_pool_layers=2, chans=64).to(device)
#netG = UnetGenerator(2, 2, 6, ngf=64)




init_weights(netG, init_type='normal',init_gain=0.01)
netG = netG.float().to(device)
norm = nn.L1Loss().to(device)
# Loss and optimizer
learning_rate = 4e-5
#loss_G = nn.MSELoss()
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
#optimG = torch.optim.RMSprop(netG.parameters(),lr = learning_rate)
train_loss = []
vali_loss = []


def CG(output, tol ,L, smap, mask, alised_image):
    return networks.CG.apply(output, tol ,L, smap, mask, alised_image)

def PSNR(img1, img2):
    #D = np.array(img2, dtype=np.float32) - np.array(img1, dtype=np.float32)
    #D[:, :] = D[:, :]**2
    #MSE = D.sum()/img1.sizev
    MSE = np.mean(np.abs(img1-img2)**2)
    psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #psnr = 10*math.log10(float(1.**2)/MSE)
    return psnr





PSNR_list =[]
train_loss= []
vali_loss =[]

#exp_lr_scheduler = optim.lr_scheduler.StepLR(optmizer,step_size= 7,gamma =0.1)
#scheduler = torch.optim.lr_scheduler.StepLR(optimG,
#                                            step_size=100, gamma=0.9)

number_of_image = 30
#learning_rate = 4e-5

#l1_penalty = torch.nn.L1Loss(size_average=False)
#reg_loss = 0
#for param in model.parameters():
#→reg_loss += l1_penalty(param)

for epoch in range(600):
    loss_G_train = 0
   # for direct, target,smap,mask in two_channel_dataset_8accelration.train_loader:
    for direct, target,smap,mask in two_channel_dataset_4accleration.train_loader:    
        input = direct.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        label = target.to(device).float()
        temp = input
        for ii in range(6):
            output = netG(temp)
            output2 = CG(output, tol =0.00005,L= 1, smap=smap, mask= mask, alised_image= input)
            temp =output2
        optimG.zero_grad()
        loss_G = fn(temp, label)#+0.01*norm(temp,label)
        loss_G.backward()
        optimG.step()
        loss_G_train += float(loss_G)  # torch.norm(label-output)/torch.norm(label)
    with torch.no_grad():
       # for vali_direct, vali_target,vali_smap,vali_mask in two_channel_dataset_8accelration.test_loader:
        for vali_direct, vali_target,vali_smap,vali_mask in two_channel_dataset_4accleration.test_loader:
            vali_input = vali_direct.to(device).float()
            vali_smap = vali_smap.to(device).float()
            vali_mask = vali_mask.to(device).float()
            vali_label = vali_target.to(device).float()
            vali_temp = vali_input
            for jj in range(6):
                vali_output = netG(vali_temp)
                vali_output2 = CG(vali_output, tol =0.00005,L= 1, smap=vali_smap, mask= vali_mask, alised_image= vali_input)
                vali_temp = vali_output2
            val_loss = fn(vali_temp,vali_label)
        local_gt = vali_label
        local_rec = vali_temp
        local_gt =convert_2chan_into_abs(local_gt).cpu().detach().numpy()
        local_rec =convert_2chan_into_abs(local_rec).cpu().detach().numpy() 
        PSNR_rec = PSNR(np.abs(local_gt),np.abs(local_rec))
    train_loss.append(loss_G_train/number_of_image)
    vali_loss.append(val_loss.item())
    PSNR_list.append(PSNR_rec)
    print('V Loss', val_loss.item())
    print(loss_G_train)
    print(epoch)
    print(PSNR_rec)


    
    
#torch.save(netG.cpu(),'Unet_global_model_data_consistance_image_200iteration.pt')





np.save(os.path.join('..','MRI_sampling','4_accerlation_result_local','local_vali_target_4_fold_1074_oracm_i7.npy'),vali_label.cpu().detach().numpy())
np.save(os.path.join('..','MRI_sampling','4_accerlation_result_local','local_vali_image_4_fold_1074_oracm_i7.npy'),vali_temp.cpu().detach().numpy())
np.save(os.path.join('..','MRI_sampling','4_accerlation_result_local','local_vali_noise_4_fold_1074_oracm_i7.npy'),vali_input.cpu().detach().numpy())










np.save(os.path.join('..','MRI_sampling','4_accerlation_result_local','local_vali_loss_8_fold_1093_orac_5.npy'),vali_loss)
np.save(os.path.join('..','MRI_sampling','4_accerlation_result_local','local_vali_train_loss_8_fold_1093_orac_5.npy'),train_loss)
np.save(os.path.join('..','MRI_sampling','4_accerlation_result_local','local_vali_PSNR_8_fold_1093_orac_5.npy'),PSNR_list)










