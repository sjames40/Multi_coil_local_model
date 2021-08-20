% This code generates Blind Learning-based reconstructions from
% undersampled MR measurements as part of the Blind Primed Supervised (BLIPS)
% Learning protocol:
%  A. Lahiri, G. Wang, S. Ravishankar, J. A. Fessler, “Blind Primed
%  Supervised (BLIPS) Learning for MR Image Reconstruction"
%	arXiv:2104.05028 [eess.IV]
%
%
% Input
% (source directory with) .mat file(s) consisting of the following variables:
% I1:       Ground Truth SoS reconstruction (nx x ny) [Used only for performance metrics. Not required during testing]
%  y:       Raw k-space data (nx*ny x ncoils) [must be zero filled for test cases]
%  S:       Coil sensitivity maps (nx x ny x ncoils)
% Q1:       Undersampling mask (nx x ny)
%
% Output:
% (at target directory) .mat file(s) consisting of the following variables:
% I1:       Ground Truth SoS reconstruction (nx x ny) [Used only for performance metrics. Not required during testing]
%  y:       Raw k-space data (nx*ny x ncoils) [must be zero filled for test cases]
%  S:       Coil sensitivity maps (nx x ny x ncoils)
% Q1:       Undersampling mask (nx x ny)
% IOut:     SOUP-DIL reconstruction
% paramsout:performance metrics, dictionaries etc. when required/requested 


%% Load files to reconstruct
addpath('./multicoil_utils/')
addpath('./utils/')
curr_pth= pwd;
addpath(curr_pth)

dir_name = 'KneeData_multicoil/'; % source directory 
outdir = 'KneeData_DLMRI_out_pe_rnd'; % target directory

cd(['/home/anishl/' dir_name]) % directory for training/test data
f = uigetfile('*.mat', 'Select Multiple Files', 'MultiSelect', 'on' ); %load list of files in directory
% load('f_list.mat') % when file list is present
%% Parameters for undersampling mask generation
acc= 5; % approx. undersampling
alpha = 2;
acs = 23; % acs lines
rng_flag=-1;
dir=2; % pe along columns
mute = 1;

%% Loop for SOUP-DIL reconstruction 

for ix = 1:1%length(f)
    load(f{ix})
    Q1 = generate_mask_alpha(size(I1),acc,alpha,dir,acs,rng_flag,mute); % generate undersampling mask
    fprintf([f{ix} '\n'])
    subbatchMRIreconstruction_multicoil % SOUP-DIL recontruction from undersampled measurements
%     save(['../' outdir '/out_' f{ix}], 'IOut','Q1','y','S', 'I1','paramsout')% save reconstruction
end
