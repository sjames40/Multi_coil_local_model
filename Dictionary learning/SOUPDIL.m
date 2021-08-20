function [D,X,obj]= SOUPDIL(Y,D0,X0,l2,numiter,optn,optn2,ct)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function for SOUP Dictionary Learning

%This is an implementation of the learning algorithms used in the simulations in the following papers:
%1) S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, �Sum of Outer Products Dictionary Learning for Inverse Problems,�
%   in IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 1142-1146.
%2) S. Ravishankar, R. R. Nadakuditi, and J. A. Fessler, �Efficient Sum of Outer Products Dictionary Learning (SOUP-DIL) 
%   and Its Application to Inverse Problems,� in IEEE Transactions on Computational Imaging, vol. 3, no. 4, pp. 694-709, Dec. 2017.

%Inputs
%Y: data matrix with training signals as columns
%D0: initial dictionary
%X0: initial coefficients matrix (sparse matrix)
%l2: parameter for sparsity penalty during learning - the penalty weight is l2^2 for L0 dictionary learning 
%    and l2 in L1 dictionary learning
%numiter: number of iterations of block coordinate descent minimization
%optn: if set to 0, function performs L0 dictionary learning, else it does
%      L1 dictionary learning (see Sadeghi et al., 2014 for OS-DL)
%optn2: when the atom update solution is nonunique, optn2=0 sets the atom to the first column of the identity 
%       matrix. Else it is set to a random unit norm vector.
%ct: if set to 1, the objective value is computed in each iteration
%the parameter L (upper bound on coefficient magnitudes in X) for L0 dictionary learning is assumed very large (inactive)

%Outputs
%D: final dictionary
%X: final sparse coefficients matrix
%obj: vector of objective function values (set to 0 unless ct is set to 1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

obj=zeros(1,numiter);
D=D0;X=X0; %initial values
[n,K]=size(D); %dictionary size
Z=eye(n);Z=Z(:,1); 

%set the (index) order for cycling over dictionary atoms/coefficents in algorithm
sd=1:K; 
%sd=randperm(K); %alternative: random order

%block coordinate descent iterations
for i=1:numiter

   %compute before updates 
   ZP=D'*Y;
   [colx,rowx,vx] = find(X');
   
   %cycle over sparse coding and dictionary atom updates
   for j=sd
       
    ind2=colx(rowx==j);
    gamm=(vx(rowx==j))';
    Dj2=D(:,j);
 
    %sparse coding     
    h=-(Dj2'*D)*sparse(X);
    h(ind2)=h(ind2)+ gamm;
    h = ZP(j,:) + h;
    if(optn==0)
    h= h.*(abs(h)>=l2); %hard-thresholding to find jth row of X
    else
    ll=l2/2;h= (sign(h)).*((abs(h)>=ll).*(abs(h)-ll)); %soft-thresholding to find jth row of X
    end
    ind=find(h);

    %dictionary atom update
    if(~any(h))
    if(optn2==0)
    Dj2=Z; %dictionary atom setting when the atom update solution is nonunique.
    else
    %alternative setting to a random unit norm vector.
    Dj2=randn(n,1);
    Dj2=Dj2/norm(Dj2,2);
    end
    else
    [ind3,inna,~]=intersect(ind2,ind);
    Dj2=(Y(:,ind)*(h(ind)')) - D*(full(X(:,ind))*(h(ind)')) + Dj2*(gamm(:,inna)*(h(ind3)')); 
    Dj2=Dj2/norm(Dj2,2); %update jth column of D
    end
    
    X(j,ind2)=0; X(j,ind)=h(ind); D(:,j)=Dj2;
   end
   
   %evaluate objective function for dictionary learning
   if(ct==1)
   if(optn==0)
   obj(i)= ((norm(Y-(D*X),'fro'))^2) + ((l2)^2)*(sum(sum(abs(X)>0)));
   else
   obj(i)= ((norm(Y-(D*X),'fro'))^2) + (l2)*(sum(sum(abs(X))));
   end
   end
end