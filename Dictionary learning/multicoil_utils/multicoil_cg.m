function [x_out,cost] = multicoil_cg(x_in,y,S,Q1,x0,nu,niter,npix)
% This function uses NCG to perform the multi-coil data update on the Super-BReD image
% using the undersampled measurements, the coil sensitivities and knowedge
% of the samplimg pattern.
%                                                                          
% inputs: 
% x_in : output of the Super-Bred algorithm (or x_DL)
% y : measured k-space data from all coils (coil dim = 2);
% S : complex sensitivity maps for all coils (coil_dim = 3)
% Q1 : undersampling pattern
% x0 : initialization for the updated image
% nu : weight associated with data-fidelity
% niter : number of iterations of NCG.
% npix : number of considered patches in image x
%
% outputs:
% x_out : updated image
%
% Anish Lahiri, Dec 2019

% x = zeros(size(x0)); % for initial image of all zeros
x = (x0);
x_noreg = x;

% to remove tikhonov regularization
% x_in = zeros(size(x_in));
% npix = 0; 

% PSNR = [];
% PSNR_noreg = [];
% [aa,bb] = size(x);


nCoils = size(S,3);
x_zf_i = zeros([size(x_in) nCoils]);

parfor i = 1:nCoils
    x_zf_i(:,:,i) = system_mtx_bwd(y(:,i),S(:,:,i),Q1);
end
x_zf_sum = sum(x_zf_i,3);

r = (nu*x_zf_sum) + x_in - (nu*AtransA_sum(x,S,Q1)) - (npix*x);
p = r;
rtr_old = dot(r(:),r(:));

% % without learned dictionary or regularization
% r_noreg = (nu*x_zf_sum) - (nu*AtransA_sum(x,S,Q1));
% p_noreg = r_noreg;
% rtr_noreg_old = dot(r_noreg(:),r_noreg(:));

% cost = zeros(niter,1);

for i =1:niter
    Ap = (nu*AtransA_sum(p,S,Q1)) + (npix*p);
    alpha = rtr_old/real(dot(p(:),Ap(:)));
    
%     % without learned dictionary or regularization
%     Ap_noreg = (nu*AtransA_sum(p_noreg,S,Q1));
%     alpha_noreg = rtr_noreg_old/(dot(p_noreg(:),Ap_noreg(:)));
    
    x = x + (alpha*p);
    r = r - (alpha*Ap);
%     figure(3);
%     subplot(121);imagesc(abs(x)); colormap hot; colorbar;axis image;axis off;
%     subplot(122);imagesc(abs(I1)); colormap hot; colorbar;axis image;axis off
%     % without learned dictionary or regularization
%     x_noreg = x_noreg + (alpha_noreg*p_noreg);
%     r_noreg = r_noreg - (alpha_noreg*Ap_noreg);
    
    
    rtr_new = dot(r(:),r(:));
    
%     % without learned dictionary or regularization
%     rtr_noreg_new = dot(r_noreg(:),r_noreg(:));
    
    if sqrt(rtr_new) < 1e-30
              break;
    end
    
    p = r + (rtr_new*p/rtr_old);
    
%     % without learned dictionary or regularization
%     p_noreg = r_noreg + (rtr_noreg_new*p_noreg/rtr_noreg_old);
    
    rtr_old = rtr_new;
    
%     % without learned dictionary or regularization
%     rtr_noreg_old = rtr_noreg_new;
    
%     if mod(i,5)==0
%         PSNR=[PSNR 20*log10((sqrt(aa*bb))*max(abs(I1(:)))/norm(double(abs(x))-double(abs(I1)),'fro'))];
%         PSNR_noreg=[PSNR_noreg 20*log10((sqrt(aa*bb))*max(abs(I1(:)))/norm(double(abs(x_noreg))-double(abs(I1)),'fro'))];
%         figure(1);
%         subplot(221);imagesc(abs(x));colormap hot; colorbar;axis off;axis image; title('image with reg')
%         subplot(223);imagesc(abs(x-I1));caxis([0 0.05*1e-7]);colormap hot;axis off;axis image;colorbar;title('error w/ reg')
%         subplot(224);imagesc(abs(x_noreg-I1));caxis([0 0.05*1e-7]);colormap hot;axis off;axis image;colorbar;title('error w/o reg')
%         subplot(222);plot(5*(0:length(PSNR)-1),PSNR); xlabel('iter');ylabel('PSNR'); axis square;
%         hold on;plot(5*(0:length(PSNR_noreg)-1),PSNR_noreg); legend({'w/ reg', 'w/o reg'},'Location','southwest'); hold off;
%         drawnow();
%     end
%     figure(2)
%     cost_p1 =  (nu*AtransA_sum(x,S,Q1)) + (npix*x);
%     cost_p2 =  ( (nu*x_zf_sum)  + x_in );
%     cost(i) =  real((0.5*dot(x(:),cost_p1(:))) - dot(x(:),cost_p2(:)));
%     plot(cost(1:i)); xlabel('iter'), ylabel('cost')
%     drawnow();
     
end
    
x_out = x;



    