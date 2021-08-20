function y_out = system_mtx_fwd(x,S,Q1)
% implements the forward pass of the system matrix operation for multi-coil
% MRI. i/p is in image domain whereas, the output is undersampled k-space.
% Essentially applies:
% 
%  y = (PFS)*x;
%
% Inputs :
% x  : complex image
% S  : complex sensitivity map corresponding to the image
% Q1 : mask applied in k-space
%
% Outputs :
% y_out : undersampled k-space
%
% Anish Lahiri, Dec 2019

Sx = S.*x; % applying sensitivity matrix 
y_full = fftshift(fft2(ifftshift(Sx))); % image to k-space
y_out = y_full(Q1==1); % under-sampling

