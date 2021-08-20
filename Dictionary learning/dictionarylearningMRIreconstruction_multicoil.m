function [IOut,paramsout] = dictionarylearningMRIreconstruction_multicoil(I1,Q1,S,y,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is an implementation of SOUP dictionary learning based magnetic resonance image reconstruction 
%used in the simulations of the following papers:
%1) S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, �Sum of Outer Products Dictionary Learning for Inverse Problems,�
%   in IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 1142-1146.
%2) S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, �Efficient Sum of Outer Products Dictionary Learning (SOUP-DIL) 
%   and Its Application to Inverse Problems,� in IEEE Transactions on Computational Imaging, vol. 3, no. 4, pp. 694-709, Dec. 2017.
%3) S. Ravishankar, B. E. Moore, R. R. Nadakuditi, and J. A. Fessler, �Efficient Learning of Dictionaries with Low-rank Atoms,�
%   in IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 222-226.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs: 1) I1 : Complex-valued reference image obtained from fully-sampled k-space data (If only undersampled measurements are available, please provide 
%                the zero-filling reconstruction as the reference. The sampling mask will then be applied in the k-space space of the zero-filled result.)
%        2) Q1 : Sampling Mask for k-space
%        3) params: Structure containing the parameters for the simulation. The various fields are as follows:
%                   - nu: weight on the data fidelity term in the problem formulation
%                   - num: number of outer iterations of the image reconstruction algorithm
%                   - num2: number of iterations of block coordinate descent in dictionary learning
%                   - n: total number of pixels in a square patch
%                   - K: number of atoms or columns in learned dictionary
%                   - et: vector that determines the sparsity penalty weight during the dictionary learning step. Each vector
%                         entry corresponds to a dictionary learning step or one outer iteration of the image reconstruction algorithm.
%                   - r: patch overlap stride (this implementation works with r=1; otherwise the quantity in Line 153 should be changed accordingly)
%                   - D0: initial dictionary in the algorithm
%                   - cini: if set to 1, the initial reconstruction in the algorithm is set to the zero-filling reconstruction. 
%                           For all other values of `cini', an external input for the initial reconstruction (estimate) is used (see next parameter). 
%                   - initrecon: An initial image reconstruction. Please ensure that this image has intensities (magnitudes) approximately in the range [0 1].
%                   - optn: set to 0 to simulate dictionary learning with L0 sparsity penalty. Set to 1 for L1 penalty.
%                   - optn1: set to nonzero value for performing DINO-KAT learning
%                   - np: vector with two entries containing the row and column dimensions of reshaped 
%                         dictionary columns (used when params.optn1 is nonzero)
%                   - p: maximum allowed non-zero rank per atom (used when params.optn1 is nonzero)
%                   - optn2: when the sparse coefficients corresponding to an atom during learning are zero, params.optn2=0 sets
%                            the atom to the first column of the identity matrix. Else it is set to an appropriate random vector.
%                   - ct: If set to 1, the code additionally outputs various performance metrics computed over the reconstruction algorithm's iterations.
%                   - co: If set to 1, the code additionally outputs various algorithm convergence metrics computed over the reconstruction algorithm's iterations.


%Outputs:  1) IOut: Reconstructed MR image.
%          2) paramsout - Structure containing various outputs, and convergence or performance metrics 
%                         for the SOUPDIL reconstruction algorithm. Many of these are vectors (whose entries correspond to values at each iteration).
%                 - dictionary: Final learnt dictionary
%                 - PSNR0: PSNR of initial reconstruction (output only when ct is set to 1)
%                 - PSNR: PSNR of the reconstruction at each iteration of the algorithm (output only when ct is set to 1)
%                 - HFEN: HFEN of the reconstruction at each iteration of the algorithm (output only when ct is set to 1)
%                 - runtime: total execution time for the algorithm (output only when ct is set to 1)
%                 - itererror: norm of the difference between the reconstructions at successive iterations (output only when co is set to 1)
%                 - obj: objective function values at each iteration of algorithm (output only when co is set to 1)
%                 - dl: dictionary learning fitting error (computed over all patches) at each iteration of algorithm (output only when co is set to 1)
%                 - dfit: value of the data fidelity component of the objective at each algorithm iteration (output only when co is set to 1)
%                 - spenalty: value of the sparsity penalty component of the objective at each algorithm iteration (output only when co is set to 1)
%                 - sparsity: fraction of nonzero coefficients (of patches) at each algorithm iteration (output only when co is set to 1)
%
% edited by Anish Lahiri for BLIPS Reconstruction:
%  A. Lahiri, G. Wang, S. Ravishankar, J. A. Fessler, “Blind Primed
%  Supervised (BLIPS) Learning for MR Image Reconstruction"
%	arXiv:2104.05028 [eess.IV]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initializing algorithm settings
[aa,bb]=size(I1);
nu=(params.nu)*(aa*bb); 
niter_cg = params.niter_cg;
num=params.num;
num2=params.num2;
n=params.n;
K=params.K;
et=params.et;
wd=params.r;
D=params.D0;
cini=params.cini;
optn=params.optn;
optn1=params.optn1;
optn2=params.optn2;
if(optn1~=0)
np=params.np;
p=params.p;
end
ct=params.ct; 
co=params.co;


nCoils = size(S,3);

% undersampling
y = y(logical(Q1(:)),:);


% zero-filled coil images
x = zeros([size(I1),nCoils]); 
parfor i = 1:nCoils
    xi = system_mtx_bwd(y(:,i),S(:,:,i),Q1)/numel(Q1);
    x(:,:,i) = xi;
end

% StransS_sum = sum(conj(S).*S,3); % not necessary for normalized coil sensitivity maps
I11 = sum(x,3); % initial average of coil images (so that E[\hat{x}] = x)
% I11 = sum(x,3)./StransS_sum; % initial average of coil images (so that E[\hat{x}] = x)
% I11(isnan(I11)) = 0;
%%%

if(cini==1)
% set to nothing because undersampled k-space is use to generate intial
% image
else
I11=params.initrecon;  %input initial image
end

I11p=I11;

%initializing performance and convergence metrics
if(ct==1)
ittime=zeros(1,num);HFEN=zeros(1,num);PSNR=zeros(1,num);
end
if(co==1)
obj=zeros(1,num); dl=zeros(1,num);dfit=zeros(1,num);spenalty=zeros(1,num);itererror=zeros(1,num);
sparsity=zeros(1,num);
end



%SOUP-DIL image reconstruction algorithm iterations
for kp=1:num 
%     tic
    
    Iiter=I11;
    
    %create image patches including wrap around patches
    Ib= [I11 I11(:,1:(sqrt(n)-1));I11(1:(sqrt(n)-1),:) I11(1:(sqrt(n)-1),1:(sqrt(n)-1))];
    [TE,idx] = my_im2col(Ib,[sqrt(n),sqrt(n)],wd);
    N2=size(TE,2); %total number of image patches
    
    if(kp==1)
        X=sparse(zeros(K,N2));
        D=params.D0;
        [rows,cols] = ind2sub(size(Ib)-sqrt(n)+1,idx);
    end

    if(optn1==0)
        [D,X]= SOUPDIL(TE,D,X,et(kp),num2,optn,optn2,0);
    else
        [D,X]= DINOKAT(TE,D,X,et(kp),np,p,num2,optn,optn2,0);
    end
    
    
%     toc
    %image update
    wt = zeros(size(Ib));
    IMoutR=zeros(size(Ib));
    IMoutI=zeros(size(Ib));
    bbb=sqrt(n);
    ZZ= (D*X);
    Lt=10000;
    for jj = 1:Lt:N2
        jumpSize = min(jj+Lt-1,N2);
        block=reshape(real(ZZ(:,jj:jumpSize)),bbb,bbb,jumpSize-jj+1);blockc=reshape(imag(ZZ(:,jj:jumpSize)),bbb,bbb,jumpSize-jj+1);
        for ii  = jj:jumpSize
            col = cols(ii); row = rows(ii);
            IMoutR(row:row+bbb-1,col:col+bbb-1)=IMoutR(row:row+bbb-1,col:col+bbb-1)+block(:,:,ii-jj+1);
            IMoutI(row:row+bbb-1,col:col+bbb-1)=IMoutI(row:row+bbb-1,col:col+bbb-1)+blockc(:,:,ii-jj+1);
            wt(row:row+bbb-1,col:col+bbb-1)=wt(row:row+bbb-1,col:col+bbb-1)+1;
        end
    end
    
    IMout=(IMoutR + (0+1i)*IMoutI);
    
%     IMout=(IMoutR + (0+1i)*IMoutI)./wt; % comment out for implicit control through data fidelity regularizer
    
    IMout2=zeros(aa,bb);
    IMout2(1:aa,1:bb)=IMout(1:aa,1:bb);
    IMout2(1:(sqrt(n)-1),:)= IMout2(1:(sqrt(n)-1),:)+ IMout(aa+1:size(IMout,1),1:bb);
    IMout2(:, 1:(sqrt(n)-1))=IMout2(:, 1:(sqrt(n)-1)) + IMout(1:aa,bb+1:size(IMout,2));
    IMout2(1:(sqrt(n)-1),1:(sqrt(n)-1))= IMout2(1:(sqrt(n)-1),1:(sqrt(n)-1))+ IMout(aa+1:size(IMout,1),bb+1:size(IMout,2));
  
    
    
    [I11] = multicoil_cg(IMout2,y,S,Q1,Iiter,nu(kp),niter_cg,n);
    
%     time=toc;
    
    %Compute various performance or convergence metrics   
    if(ct==1)
%     ittime(kp)=time;
    PSNR(kp)=20*log10((sqrt(aa*bb))*max(abs(I1(:)))/norm(double(abs(I11))-double(abs(I1)),'fro'));
    HFEN(kp)=norm(imfilter(abs(I11),fspecial('log',15,1.5)) - imfilter(abs(I1),fspecial('log',15,1.5)),'fro');
    end
   
  
end

%Outputs
IOut=I11;

% paramsout.dictionary=D;
% paramsout.PSNR0=20*log10((sqrt(aa*bb))*max(abs(I1(:)))/norm(double(abs(I11p))-double(abs(I1)),'fro'));
% paramsout.PSNR = 20*log10((sqrt(aa*bb))*max(abs(I1(:)))/norm(double(abs(I11))-double(abs(I1)),'fro'));

if(ct==1)
    paramsout.PSNR0=20*log10((sqrt(aa*bb))*max(abs(I1(:)))/norm(double(abs(I11p))-double(abs(I1)),'fro'));
    paramsout.PSNR=PSNR;
    paramsout.HFEN=HFEN;
%     paramsout.runtime=sum(ittime);
    paramsout.I11p = I11p;
    paramsout.D =D;
end

if(co==1)
    paramsout.itererror=itererror;
    paramsout.obj=obj;
    paramsout.dl=dl;
    paramsout.dfit=dfit;
    paramsout.spenalty=spenalty;
    paramsout.sparsity=sparsity;
end