function x_out = AtransA_sum(x_in,S,Q1)
nCoils = size(S,3);
x_out_s = zeros([size(x_in),nCoils]);
parfor i = 1:nCoils
x_out_s(:,:,i) = system_mtx_bwd(system_mtx_fwd(x_in,S(:,:,i),Q1),... 
    S(:,:,i),Q1);
end
x_out = sum(x_out_s,3);