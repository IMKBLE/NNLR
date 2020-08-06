accuracyMatMean = mean(accuracyMat,2);
accuracyMatStd=zeros(6,1);
for i=1:6
    accuracyMatStd(i,1) = std(accuracyMat(i,:));
end
