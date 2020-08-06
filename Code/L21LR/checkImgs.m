for i=1:28
    img = reshape(fea(:,i),50,40);
    subplot(4,7,i),imshow(img,[]);
end