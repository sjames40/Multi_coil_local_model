import numpy as np
from bart import bart
import h5py
import sys
import os


rt = '/mnt/DataA/multicoil_val' # data to load
aim = '/mnt/DataA/KSPACE' # aim folder
os.chdir(rt)
foldername = []

for i in os.listdir():
    if i.endswith('h5'):
        foldername.append(i)
foldername.sort()
arr = os.listdir(rt)

def make_two_channel_kspace_smap(aim,rt):
    hf = h5py.File(os.path.join(filename,filename1), 'r')
    kspace = hf['kspace'][()]
    num_layers, num_coils, nx, ny = kspace.shape
    kmax = np.amax(np.abs(kspace[:]))
    for i in range(num_layers):
        k = kspace[i,:,:,:]
        k_r = (np.real(k)/kmax*32767).astype('int16') # save memory, should be consistent with your dataloader!
        k_i = (np.imag(k)/kmax*32767).astype('int16')
        
        k = np.expand_dims(k, axis=0)
        k = np.transpose(k,(2,3,0,1))
        try:
            sens_maps = bart(1, f'ecalib -d0 -m1 -r24 -k6 -t0.001 -c0.95', k)
            print(sens_maps)
        except:
            print('oops')
            continue
        sens_map = sens_maps.squeeze().transpose((2,0,1))
        s_r = (np.real(sens_map)*32767).astype('int16')
        s_i = (np.imag(sens_map)*32767).astype('int16')
        np.savez(os.path.join(aim,'%s_Layer%d.npz'%(filename,i)),s_r=s_r,s_i=s_i,k_r=k_r,k_i=k_i)




        
        
        
        
        
        
        
        

        
