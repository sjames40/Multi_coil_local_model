import torch

import torch.nn as nn
import torch.nn.functional as F
#import dataset2
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
from torch.nn import init

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
#netG = Unet(in_chans=2,
#            out_chans=2).to(device)
netG =Unet(2,2, num_pool_layers=2, chans=32).to(device)
#netG = UNet().to(device)
init_weights(netG, init_type='normal',init_gain=0.02)
netG = netG.float()
norm =  nn.L1Loss().to(device)
#tst=torch.zeros(nEpochs)
#vtst=torch.zeros(nEpochs)
# Loss and optimizer
learning_rate = 7e-5
loss_G = nn.MSELoss()
fn = nn.MSELoss().to(device)
optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
train_loss =[]
vali_loss =[]
def NormalizeData2(data):
    return data/torch.max(torch.abs(data))

for epoch in range(500):
    loss_G_train = 0
    for direct, target in two_channel_dataset_test.train_loader:
        input = direct.to(device).float()
        # print(input.shape)
        #input = NormalizeData2(input) 
         # input =torch.abs(input)
        label = target.to(device).float()
        #label = NormalizeData2(label)
        output = netG(input)
        # print(output.shape)    

         # backward netG
        optimG.zero_grad()
         #if bfov: loss_G = fn(output * mask, label * mask)
        loss_G = fn(output, label)+ 0.1*norm(output,label)
       # loss_G = fn(input,label)+0.1*fn(output,label)
        loss_G.backward()
        optimG.step()
        loss_G_train += loss_G.item()#torch.norm(label-output)/torch.norm(label)
    with torch.no_grad():
        for vali_direct,vali_target in two_channel_dataset_test.test_loader:
            vali_input =  vali_direct.to(device).float()
            val_pred = netG(vali_input)
            vali_target = vali_target.to(device).float()    
            val_loss = fn(val_pred, vali_target)

    train_loss.append(loss_G_train)
    #vali_loss.append(val_loss.item())
    print('V Loss', val_loss.item())
    print(loss_G_train)
    print(epoch)

    


    
    
#torch.save(netG.cpu(),'Unet_global_model_data_two_channel_correct.pt')




    
    
    
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_target_4_fold_1082_2.npy'),vali_target.cpu().numpy())
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_image_4_fold_1082_2.npy'),val_pred.cpu().numpy())
np.save(os.path.join('..','MRI_descattering','two_channel_result','vali_noise_4_fold_1082_2.npy'),vali_input.cpu().numpy())


#np.
save(os.path.join('..','MRI_descattering','train_noise_early_datasset_0.1L1_25noisy.npy'),input.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','train_image_early_datasset_0.1L1_25noisy.npy'),output.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','train_gh_early_datasset_0.1L1_25noisy.npy'),label.cpu().detach().numpy())    
np.save(os.path.join('..','MRI_descattering','val_gh_4_fold_data_consistance.npy'),vali_loss)

np.save(os.path.join('..','MRI_descattering','train_gh_data_consistance.npy'),train_loss) 











    
    
    
    
