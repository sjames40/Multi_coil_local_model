function psnr = psnrfun(I1,IOut)

psnr = 20*log10((sqrt(numel(I1)))*max(abs(I1(:)))/norm(double(abs(IOut))-double(abs(I1)),'fro'));

end