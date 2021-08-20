%This demo code is based on SOUP dictionary learning based image reconstruction in the following papers:
%1) S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, �Sum of Outer Products Dictionary Learning for Inverse Problems,�
%   in IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 1142-1146.
%2) S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, �Efficient Sum of Outer Products Dictionary Learning (SOUP-DIL) 
%   and Its Application to Inverse Problems,� in IEEE Transactions on Computational Imaging, vol. 3, no. 4, pp. 694-709, Dec. 2017.

% edited by Anish Lahiri for BLIPS Reconstruction:
%  A. Lahiri, G. Wang, S. Ravishankar, J. A. Fessler, “Blind Primed
%  Supervised (BLIPS) Learning for MR Image Reconstruction"
%	arXiv:2104.05028 [eess.IV]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc
% clear
rng(0)


num_conv = 40;
%set parameters for dictionary learning based MR image reconstruction
[aa,bb]=size(I1);
params.niter_cg = 5;
params.num=20;%20 % number of outer iterations of the image reconstruction algorithm
params.num2=5; % number of iterations of block coordinate descent in dictionary learning
params.n=36; % number of pixels in patch
params.K=4*params.n; %number of atoms or columns in dictionary
wt1 = 1e-4; wt2 = 8e-4;% 8e-4
params.nu=[logspace(log10(wt1), log10(wt2),params.num-num_conv) wt2*ones(1,num_conv)]/(aa*bb);
a=1.0; 
a2=0.2;
params.et=1e-7*[logspace(log10(a), log10(a2),params.num-num_conv) a2*ones(1,num_conv)]; %determines sparsity penalty weight in dictionary learning step.
%Each entry of "et" corresponds to one outer iteration (or one learning step) of the reconstruction algorithm.

params.r=1; %patch overlap stride

D0 = genODCT1(params.n,params.K,1,1);

params.D0=D0; %initial dictionary
params.cini=1; %sets the initial reconstruction in the algorithm to a zero-filling reconstruction
params.optn=0; %simulates dictionary learning with L0 sparsity penalty
params.optn1=0; %not performing DINO-KAT learning
params.optn2=0; %when the sparse coefficients corresponding to an atom during learning are zero, params.optn2=0 
                %sets the atom to the first column of the identity matrix. Else it is set to an appropriate random vector.
params.ct=1; %output performance metrics computed over algorithm iterations
params.co=0; %do not output convergence metrics computed over algorithm iterations
params.nl=0; %indicates peak intensity value in the reference image is 1

%MR image reconstruction with SOUP dictionary learning
[IOut,paramsout] = dictionarylearningMRIreconstruction_multicoil(I1,Q1,S,y,params);