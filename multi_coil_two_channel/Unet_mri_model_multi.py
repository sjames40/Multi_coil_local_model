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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

nChannels = 1
nClasses=1
nEpochs =30



clean_data = []
clean_data_name = '/home/liangs16/labmat_project/MRI_descattering/multi_coil_8_fold_GT/'
noisy_data_name = '/home/liangs16/labmat_project/MRI_descattering/multi_coil_8_fold_RC/'
noisy_data = []
noisy_array = os.listdir(noisy_data_name)
clean_array = os.listdir(clean_data_name)
noisy_array.sort()

clean_array.sort()
#print(clean_array)
#for i in range(9,10): #13,14 11,12
#    print(i)
#    noisy_file = noisy_array[i]
#    clean_file = clean_array[i]
#    clean_data_from_file = np.load(os.path.join(clean_data_name,clean_file),'r')
#    noisy_data_from_file = np.load(os.path.join(noisy_data_name,noisy_file),'r')
#    noisy_data.append(noisy_data_from_file['arr_0'][20])
#    clean_data.append(clean_data_from_file['arr_0'][20])
   # noisy_data.append(np.load(os.path.join('..','MRI_descattering','vali_image_early_datasset_0.1L1_20noisy.npy'))[0][0])

    


    


    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Unet(in_chans=2,
            out_chans=2).to(device)
#netG = UNet().to(device)
netG = netG.float()
norm =  nn.L1Loss().to(device)
#tst=torch.zeros(nEpochs)
#vtst=torch.zeros(nEpochs)
# Loss and optimizer
learning_rate = 1e-4
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
       # loss_G = fn(input,label)+0.1*fn(output,label)
        loss_G.backward()
        optimG.step()
        loss_G_train += loss_G.item()#torch.norm(label-output)/torch.norm(label)
    #with torch.no_grad():
    #    vali_input = np.abs(noisy_data[0])
    #    vali_input = torch.from_numpy(vali_input).unsqueeze(0).unsqueeze(0).to(device).float()
    #    vali_input = NormalizeData2(vali_input)
    #    val_pred = netG(vali_input)
    #    vali_target = np.abs(clean_data[0])
    #    vali_target = torch.from_numpy(vali_target).unsqueeze(0).unsqueeze(0).to(device).float()
    #    vali_target = NormalizeData2(vali_target)
    #    val_loss = fn(val_pred, vali_target)
        # for j in range(len(noisy_data)):
              #vali_input = np.abs(noisy_data[0])
              #vali_input = torch.from_numpy(vali_input).unsqueeze(0).unsqueeze(0).to(device).float()
              #vali_input = NormalizeData2(vali_input)
              #val_pred = netG(vali_input)
              #vali_target = np.abs(clean_data[0])
              #vali_target = torch.from_numpy(vali_target).unsqueeze(0).unsqueeze(0).to(device).float()
              #vali_target = NormalizeData2(vali_target)
              #val_loss = fn(val_pred, vali_target)


    train_loss.append(loss_G_train)
    #vali_loss.append(val_loss.item())
    #print('V Loss', val_loss.item())
    print(loss_G_train)
    print(epoch)

    
    
    

    
    
    


    

    
torch.save(netG.cpu(),'Unet_global_model_data_consistance_image.pt')








    
    
    
    
np.save(os.path.join('..','MRI_descattering','vali_target_4_fold_data_consistance.npy'),val_target.cpu().numpy())
np.save(os.path.join('..','MRI_descattering','vali_image_4_fold_data_consistance.npy'),val_pred.cpu().numpy())
np.save(os.path.join('..','MRI_descattering','vali_noise_4_fold_data_consistance.npy'),vali_input.cpu().numpy())
#np.save(os.path.join('..','MRI_descattering','train_noise_early_datasset_0.1L1_25noisy.npy'),input.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','train_image_early_datasset_0.1L1_25noisy.npy'),output.cpu().detach().numpy())
#np.save(os.path.join('..','MRI_descattering','train_gh_early_datasset_0.1L1_25noisy.npy'),label.cpu().detach().numpy())    
np.save(os.path.join('..','MRI_descattering','val_gh_4_fold_data_consistance.npy'),vali_loss)

np.save(os.path.join('..','MRI_descattering','train_gh_data_consistance.npy'),train_loss) 











    
    
    
    
