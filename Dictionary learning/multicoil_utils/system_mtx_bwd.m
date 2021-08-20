function x_out = system_mtx_bwd(y,S,Q1)
% implements the adjoint of the system matrix operation for multi-coil
% MRI. i/p is in k-space whereas, the output is in image domain.
% Essentially applies:
% 
%  y = (PFS)*x;
%
% Inputs :
% y  : complex k-space measurements
% S  : complex sensitivity map corresponding to the image
% Q1 : mask applied in k-space
%
% Outputs :
% x_out : zero-filled sensitized image
%
% Anish Lahiri, Dec 2019


y_zf = zeros(size(Q1));
y_zf(Q1==1) = y; % zero-filling
x_out =(numel(Q1))*ifftshift(ifft2(fftshift(y_zf)))... % ifft of ZF image
    .*conj(S);                              % adjoint of sensitivity multiplication

