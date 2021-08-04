from model import *
from dataset import *
import dataset2
import torch
import torch.nn as nn
from torch.nn import DataParallel as DDP
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from pdb import set_trace

import xarray as xr
import logging

import src.data
import src.models
import src.descatter
import src.experiments

data_name = 'multi'
#model_type = 'local_conv'

# set up a log file
#src.experiments.setup_logging('test')

# load image daota
rho, D, S, T, test, train = src.data.prep(data_name)

mode = 'train'

#train_continue = self.train_continue
num_epoch = 1000

lr_G = 7e-5
#batch_size = 24
#device = self.device
#torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#gpu_ids = self.gpu_ids

#nch_in = D.sel(train).data
#nch_out = self.nch_out
#nch_ker = self.nch_ker
#bfov = self.bfov
norm =  nn.L1Loss().to(device)
#num_freq_disp = self.num_freq_disp
#num_freq_save = self.num_freq_save




#netG = CNN_wo_skip(nch_in=1, nch_out=1, nch_ker=64, norm='bnorm')
netG = UNet(nch_in=1, nch_out=1, nch_ker=64, norm='bnorm',snorm =True)

#elif self.network_type == 'cnn_wo_skip':
#   netG = CNN_wo_skip(nch_in=self.nch_in, nch_out=self.nch_out, nch_ker=self.nch_ker, norm=self.norm)
#elif self.network_type == 'cnn_w_skip':
#   netG = CNN_w_skip(nch_in=self.nch_in, nch_out=self.nch_out, nch_ker=self.nch_ker, norm=self.norm)

netG.to(device)
init_weights(netG, init_type='normal', init_gain=0.02)

        ## setup loss & optimization
#if self.loss_type == 'l1':
#   fn = nn.L1Loss().to(device)  # Regression loss: L1
#elif self.loss_type == 'l2':
#   fn = nn.MSELoss().to(device)  # Regression loss: L2
#elif self.loss_type == 'l2_norm':
#   fn = lambda input, target: ((((input - target) ** 2) / torch.norm(target, 2, dim=(1, 2, 3), keepdim=True)).mean()).to(device)
#elif self.loss_type == 'cls':
#   fn = nn.CrossEntropyLoss().to(device)  # Classification loss: Cross entropy

paramsG = netG.parameters()
optimG = torch.optim.Adam(paramsG, lr=lr_G)

#schedG = torch.optim.lr_scheduler.MultiStepLR(optimG, [300, 600], gamma=0.1)

        ## load from checkpoints
st_epoch = 0
fn = nn.MSELoss().to(device)
#mask = 1.0 * (input > 0)
#if train_continue == 'on': # netG, optimG, st_epoch = self.load(dir_chck, netG, optimG, mode=mode)
#   netG, _, st_epoch = self.load(dir_chck, netG, optimG, mode=mode)
#if gpu_ids:
#   netG = DDP(netG, device_ids=gpu_ids)

        ## setup tensorboard
#writer_train = SummaryWriter(log_dir=dir_log_train)
#writer_val = SummaryWriter(log_dir=dir_log_val)


#optimG.zero_grad()
#optimG.step()
#schedG.step(epoch=st_epoch)
def NormalizeData2(data):
    return data/torch.max(torch.abs(data))
for epoch in range(0 + 1, num_epoch + 1):
            ## training phase
     netG.train()

     loss_G_train = 0

     for direct, target in dataset2.train_loader: 
         #def should(freq):
         #     return freq > 0 and (i % freq == 0 or i == num_batch_train)

         input = direct.to(device)
         label = target.to(device)
         #total = input + label
         #mask = 1.0*(total>0)
         #input =total
         #input = -torch.log(mask*input+(1-mask))
         input[torch.isnan(input)]=0
         input[torch.isinf(input)]=0
         norm = torch.max(torch.abs(input))
         #input = NormalizeData2(input)
         input =input/norm
        # print(input.shape)
        # label = target.to(device)
         #label = -torch.log(mask*label+(1-mask))
         label[torch.isnan(label)]=0
         label[torch.isinf(label)]=0
         #label = NormalizeData2(label)
         label =label/norm
         output = netG(input)  
        # output = torch.exp(-output) * mask
         output[torch.isnan(output)] = 0
         output[torch.isinf(output)] = 0
        
        # print(output.shape)    
                # backward netG
         optimG.zero_grad()
         #if bfov: loss_G = fn(output * mask, label * mask)
         loss_G = fn(output, label)

         loss_G.backward()
         optimG.step()
         loss_G_train += loss_G.item()
     with torch.no_grad():
         input_val = D.sel(test)[0]
         output_val = S.sel(test)[0]
     
         input_val  =transform.resize(input_val, (256, 256), anti_aliasing=True)
         input_val = torch.from_numpy(input_val).unsqueeze(0).unsqueeze(0).float().to(device)
         output_val  =transform.resize(output_val, (256, 256), anti_aliasing=True)
         output_val = torch.from_numpy(output_val).unsqueeze(0).unsqueeze(0).float().to(device)
         #total_val = input_val+output_val
         #input_val = total_val
         #mask_val = 1.0 * (input_val > 0)
         #input_val = -torch.log(mask_val*input_val+(1-mask_val))
         
         input_val[torch.isnan(input_val)]=0
         input_val[torch.isinf(input_val)]=0
         val_norm = torch.max(torch.abs(input_val))
         #input_val = NormalizeData2(input_val)
         input_val= input_val/val_norm
         pred = netG(input_val)
         #pred = torch.exp(-pred) * mask_val
         pred[torch.isnan(pred)] = 0
         pred[torch.isinf(pred)] = 0
         #pred = pred/val_norm

         #output_val = -torch.log(mask_val*output_val+(1-mask_val))
         output_val[torch.isnan(output_val)]=0
         output_val[torch.isinf(output_val)]=0
         output_val = output_val/val_norm
         #output_val = NormalizeData2(output_val)#.to(device) 
         val_loss = fn(pred, output_val)
    
     print(val_loss)
     print(loss_G_train)
     print(epoch)

     
     

#pred = torch.exp(-pred) * mask_val      
     
torch.save(netG.cpu(),'multi_3layer_local_Unet_D_to_S_for_image2_log.pt')
np.save(os.path.join('..','descattering','reports','x_local_2layer_val_RC_D_to_S20image_onenorm'+str(1)+'.npy'),pred.cpu()
        .detach().numpy())











