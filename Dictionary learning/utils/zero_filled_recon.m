function I1 = zero_filled_recon(y,S,Q1)
% zero-filled coil images

x = zeros(size(S));
nCoils = size(S,3);
for i = 1:nCoils
    xi = system_mtx_bwd(y(:,i),S(:,:,i),Q1)/numel(Q1);
    x(:,:,i) = xi;
end

StransS_sum = sum(conj(S).*S,3);
% I1 = sum(x,3); % initial average of coil images (so that E[\hat{x}] = x)
I1 = sum(x,3)./StransS_sum;
I1(isnan(I1)) = 0;
