for i = 1:34982
    figure('visible','off');
    sig = b{i};
    [WVD,t,f] = wvd(sig,fs); 
    %stft(x,fs,Window=kaiser(256,5),OverlapLength=220,FFTLength=512)
    imagesc(T,F,WVD);
    set(gca,'ColorScale','log','YDir','reverse');
    axis off;
    axis xy;
    filename = ['WVD' num2str(i) '.jpg'];
    IM = frame2im(getframe(gcf));
    ReSize = imresize(IM, [240, 240], 'bicubic');
    imwrite(ReSize,filename);
    close(gcf);
    i = i+1;
end