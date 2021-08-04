import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import numpy as np
import glob
import os
import cv2
import xarray as xr
import logging
import skimage
from skimage import transform
import src.data
import src.models
import src.descatter
import src.experiments

data_name = 'multi'
model_type = 'local_conv'
data_name2 = 'mono'
data_name_2 = 'multi_2'
data_name_3 = 'multi_3'
data_name_4 = 'multi_4' 
# set up a log file
#src.experiments.setup_logging('test')

# load image daota
#rho, D, S, T, test, train = src.data.prep(data_name)
rho, D, S, T, test, train = src.data.prep(data_name) 
#rho1, D1, S1, T1, _, _ = src.data.prep(data_name_2)
#rho2, D2, S2, T2, _, _ = src.data.prep(data_name_3)
#rho3, D3, S3, T3, _, _ = src.data.prep(data_name_4) 
#clean_data = glob.glob(os.path.join("..","residue_data","image_SRF_2_reshape_odd_number",'*.npy'))
#clean_data.sort()
#noisy_data = glob.glob(os.path.join("..","residue_data","image_SRF_2_noise_sigma_40_odd_number", '*.npy')) # to do make relative path to fix newfolder
#noisy_data.sort()
#denoised_data = glob.glob(os.path.join("..","residue_data","result_of_TFthird", '*.npy'))
#denoised_data.sort()
total_data_clean =np.zeros((490,257,257))
total_data_clean[0:490,:,:] = S.sel(train).data
#total_data_clean[500:1000,:,:] = rho1.data
#total_data_clean[1000:1500,:,:] = rho2.data
#total_data_clean[1500:1900,:,:] = rho3.data[0:400,:,:] 
clean_data = total_data_clean
total_data_noisy =np.zeros((490,257,257)) 
total_data_noisy[0:490,:,:] = D.sel(train).data                                                                               
#total_data_noisy[500:1000,:,:] = T1.data                                                                           
#total_data_noisy[1000:1500,:,:] = T2.data                                                                          
#total_data_noisy[1500:1900,:,:] = T3.data[0:400,:,:]
noisy_data = total_data_noisy
#print(clean_data[1])
clean_ref = S.sel(test).data[0]
noisy_ref = D.sel(test).data[0]
def norm(x, y):
    return np.nanmean(np.power(x - y, 2),axis=(1, 2))

num_neighbors= 30
for i in  range(1):
    distances = norm(noisy_ref, noisy_data)
    inds = np.argsort(distances)   
    x_train_local = noisy_data[inds[:num_neighbors]]
    y_train_local = clean_data[inds[:num_neighbors]]

len_data = len(x_train_local)

print(len_data)



train_size = 1
num_train = int(len_data*train_size)


#train_image_paths = denoised_data[:num_train]  # todo these aren't paths
#test_image_paths = denoised_data[num_train:] # todo: use num_train
train_noisy_paths = x_train_local[:num_train]  # todo these aren't paths
#test_noisy_paths = noisy_data[num_train:]
train_clean_paths = y_train_local[:num_train]

#test_clean_paths = clean_data[num_train:]


class CustomDataset(Dataset):
    def __init__(self,clean_paths,noisy_paths):
         #self.target_paths = target_paths # denoised_img
         self.clean_paths = clean_paths  # clean_image
         self.noisy_paths = noisy_paths # noisy_image

    def __getitem__(self, index):
        clean_im = self.clean_paths[index]
        noisy_im = self.noisy_paths[index]
     
        target =transform.resize(clean_im, (256, 256), anti_aliasing=True)
       # target[np.isnan(target)]=0
        target = torch.from_numpy(target).unsqueeze(0).float()
        direct_im =transform.resize(noisy_im, (256, 256), anti_aliasing=True)
       # direct_im[np.isnan(direct_im)]=0
        direct_im = torch.from_numpy(direct_im).unsqueeze(0).float()
        return direct_im,target

    def __len__(self):
         return len(self.clean_paths)

train_dataset = CustomDataset(train_clean_paths,train_noisy_paths)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True)

#test_dataset = CustomDataset(test_clean_paths)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

