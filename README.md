# BLIPSrecon
Code for reproducing results in the BLIPS paper:

Anish Lahiri, Guanhua Wang, Sai Ravishankar, Jeffrey A. Fessler, (2021).
"Blind Primed Supervised (BLIPS) Learning  for MR Image Reconstruction."
IEEE Transactions on Medical Imaging
http://doi.org/10.1109/TMI.2021.3093770;
[arXiv preprint arXiv:2104.05028.](https://arxiv.org/abs/2104.05028)

The code is made up of two components: 
* Blind dictionary learning (MATLAB version 2020+)
* Supervised learning (with PyTorch > 1.7.0).
* Additionally, we used [BART](https://mrirecon.github.io/bart/) to generate the dataset.


## Blind Dictionary Learning-based Image Reconstruction
The MATLAB code performs without issues on MATLAB 2020+.

Run `batchSOUP_DLMRI_randmask.m` to reconstruct images from an input directory.

The input directory should consist of `.mat` files for individual slices,
and include `I1` (ground truth), `S` (sensitivity map) and `Q1` (sampling masks).
The option to save the dictionary learning output to a directory
is available in the script upon uncommenting `line 53`.
The code saves
`IOut` (dictionary learning reconstruction),
`y` (k-space),
`I1` (ground truth), `S` (sensitivity map) and `Q1` (sampling masks),
along with `paramsout` for performace metrics as needed.

`subbatchMRIreconstruction_multicoil.m` 
specifies the parameter choices
(such as number of inner and outer iterations, sparsity penalty weight, etc.)
for dictionary learning reconstruction from multicoil MR measurements.

`dictionarylearningMRIreconstruction_multicoil.m`
generates a reconstruction from undersampled multi-coil k-space measurements
using SOUP-DIL dictionary learning-based regularization.

`SOUPDIL.m` is the inner dictionary learning and sparse coding function
that takes overlapping patches in an initial image
and learns a dictionary and a set of sparse coefficients to represent these patches.


## Deep Supervised Learning-based Image Reconstruction

The supervised learning approach used a training set generated from the
[fastMRI project](https://fastmri.org/).
* `Preprocessing.ipynb` provides an example of the data-preprocessing.
* `MODL_DLMRI_Knee_vd.sh` gives an example of training the neural network.
* The file `requirements.txt` denotes related Python packages.
