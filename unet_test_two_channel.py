import torch
import torch.nn as nn
import torch.nn.functional as F
#import dataset2

import numpy as np
import os
from src.Unet_model_fast_mri  import Unet
netG = Unet(in_chans=2,
            out_chans=2)

model = torch.load(os.path.join(r"C:\Users\liang\OneDrive\Desktop\Sample_image\Unet_two_channel_model_from_0_12_data.pt"))
rec_iamge = np.load(os.path.join(r"C:\Users\liang\OneDrive\Desktop\Sample_image\RC_Image_4_fold_2channel\
vali_image_4_fold_data_consistance.npy"))
#rec_iamge = np.load(os.path.join(r"C:\Users\liang\OneDrive\Desktop\dataset2\train_image_020_datasset0.npy"))
rec_iamge = rec_iamge[0][0]
gh_iamge = np.load(os.path.join(r"C:\Users\liang\OneDrive\Desktop\Sample_image\RC_Image_4_fold_2channel\
vali_target_4_fold_data_consistance.npy"))
gh_iamge = gh_iamge[0][0]
with torch.no_grad():
    input =  image_data3['arr_0'][20]
    input = np.abs(input)
    #input = NormalizeData2(input)
    input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).float()
    # input =input/255
    pred = model(input)
    pred = pred[0][0]
    # pred = torch.exp(-pred)*mask
    pred = pred.numpy()
    # pred = pred*255
    output =  groundtruth
    output = np.abs(output)
    output = NormalizeData2(output)
    #output = output[0][0]
    #output = output.numpy()
    print(np.count_nonzero(output))