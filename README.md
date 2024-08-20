MRI global model recon
=====================================

Code for testing and reproduicing results for MRI local study project:

Modl and blinps code according to the following paper

Anish Lahiri, Guanhua Wang, Sai Ravishankar, Jeffrey A. Fessler, (2021). "Blind
Primed Supervised (BLIPS) Learning for MR Image Reconstruction." IEEE
Transactions on Medical Imaging http://doi.org/10.1109/TMI.2021.3093770; [arXiv
preprint arXiv:2104.05028.](https://arxiv.org/abs/2104.05028)

The code is made up of three components: \* single coil two channel study \*,\*
multi coil two channel global study \*, and \*multi coil two channel local model\*(with
PyTorch \> 1.7.0). \* Additionally, we used
[BART](https://mrirecon.github.io/bart/) to generate the dataset.

MRI local learning
------------------
S. Liang, A. Lahiri and S. Ravishankar, "Adaptive 
Local Neighborhood-based Neural Networks for MR Image Reconstruction from Undersampled Data," 
in IEEE Transactions on Computational Imaging, doi: 10.1109/TCI.2024.3394770.
[https://ieeexplore.ieee.org/abstract/document/10510040]

make_two_channel_dataset specifies and show how to make the two channel dataset
based on the modification form https://github.com/JeffFessler/BLIPSrecon

 

global dataset  specifies the data loader for MRI image loading from
multicoil MR measurements for global case.

local network dataset and local netowrk dataset oracle specifies the data loader for MRI image loading from
multicoil MR measurements for noise local case and oracle local case.


train local unet  can be used for local  model training and testing
reconstruction from undersampled mulit-coil k-space measurements using UNet
training.

transfer learning local network  can be used for local  model training and testing
reconstruction from undersampled mulit-coil k-space measurements using DIDN
training.

 
-

The file `requirements.txt` denotes related Python packages.
------------------------------------------------------------
Data avaliable on https://www.dropbox.com/scl/fi/801dxovhbkp2bkl2krz5x/NEW_KSPACE.zip?rlkey=4u3b32f6c4pfujsv3kp7z5bdk&st=hwe9thrv&dl=0
just put the data on the your own dataset on the Kspace_data_name= '/mnt/DataA/NEW_KSPACE' and make the image space image based on the kspace data.
