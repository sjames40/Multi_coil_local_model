import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'notebook')
import math
import torch
import time
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import scipy.io as sio
from ismrmrdtools import show, transform
# import ReadWrapper
import read_ocmr as read


# In[2]:


class L1WaveletRecon(sp.app.App):
    def __init__(self, ksp, mask, mps, lamda, max_iter):
        img_shape = mps.shape[1:]

        S = sp.linop.Multiply(img_shape, mps)
        F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
        P = sp.linop.Multiply(ksp.shape, mask)
        self.W = sp.linop.Wavelet(img_shape)
        A = P * F * S * self.W.H

        proxg = sp.prox.L1Reg(A.ishape, lamda)

        self.wav = np.zeros(A.ishape, np.complex)
        alpha = 1
        def gradf(x):
            return A.H * (A * x - ksp)

        alg = sp.alg.GradientMethod(gradf, self.wav, alpha, proxg=proxg,
                                    max_iter=max_iter)
        super().__init__(alg)

    def _output(self):
        return self.W.H(self.wav)
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
def make_hors_mask(N1,N2,nlines,init_lines):
    # Setting Variable Density Random Sampling (VDRS) mask
    mask_vdrs=np.zeros((N1,N2),dtype='bool')
    mask_vdrs[int(0.5*(N1-init_lines)):int(0.5*(N1+init_lines)),:]=True
    nlinesout=int(0.5*(nlines-init_lines))
    rng = np.random.default_rng()
    t1 = rng.choice(int(0.5*(N1-init_lines))-1, size=nlinesout, replace=False)
    t2 = rng.choice(np.arange(int(0.5*(N1+init_lines))+1, N1), size=nlinesout, replace=False)
    mask_vdrs[t1,:]=True; mask_vdrs[t2,:]=True
    return mask_vdrs
def MRI(im, mask=None):
    """
    returns Fourier samples of im
    """

    y = torch.rfft(im, 2, onesided=False, normalized=True)  # compute a DFT
    y = y.roll(tuple(n//2 for n in y.shape[:2]), dims=(0,1))  # move the center frequency to the center of the image
    if mask is not None:  # apply a sampling mask if one is supplied (used in a later question)
        y[~mask, :] = 0
    return y


def MRI_inv(y):
    """
    returns the inverse DFT of the k-space samples in y, so that MRI_inv(MRI(im)) = im
    """

    y = y.roll(tuple(-(n//2) for n in y.shape[:2]), dims=(0,1))  # move the center frequency back to the upper left corner

    im = torch.irfft(y, 2, onesided=False, normalized=True)  # undo the DFT

    return im

# In[ ]:


file1 = open('/mnt/shared_b/data/OCMR/ocmr_cine/filenames.txt', 'r')
Lines = file1.readlines()
# Lines = Lines[1:-1]
count=0;
nFiles=53; # Total files to be read
filenames=[]

for i in range(44,45):#Lines:
    if i>=44:
        print(i)
        line=Lines[i]
        filename='/mnt/shared_b/data/OCMR/ocmr_cine/'+line
        filename = filename[:-1] # Removing newline character
        filenames.append(filename)
        print(filenames)

        count=count+1
        print('Reading file No.',count,':',filename)
        if count==nFiles: # Break if exceeds the total files
            break

        kData,param = read.read_ocmr(filename);
        print('Dimension of kData:',kData.shape)

        # kspace_dim': {'kx ky kz coil phase set slice rep avg'}

        N1=kData.shape[0]
        N2=kData.shape[1]
        Nt=kData.shape[4]
        mask = make_hors_mask(N1,N2,27,13)
        mask = torch.from_numpy(mask)
        x_GT=np.zeros((N1,N2,Nt),dtype='complex')
        for phase in range(0,Nt):
            y=np.squeeze(kData[:,:,:,:,phase,0,0,0,0]) # Fully sampled k-space data for a particular time instance
            print('Testing for phase =',phase, 'with k-space data dimension: ',y.shape)

            mask_GT=np.ones((N1,N2))

            ksp=np.moveaxis(y, -1, 0) # Reaaranging axes: Loading k-space data in sigpy format
            mps = mr.app.EspiritCalib(ksp).run()
            lamda = 0
            recon = torch.from_numpy(ksp)
            recon = torch.abs(recon)
            recon = MRI(recon,mask)
            recon = MRI_inv(recon)
            x_reconstruct[phase,:,:]= recon.numpy()

            
           # max_iter = 5

           # x_GT[:,:,phase]=L1WaveletRecon(ksp, mask_GT, mps, lamda, max_iter).run()
        np.savez('x_GT'+str(i)+'.npz',x_GT)
