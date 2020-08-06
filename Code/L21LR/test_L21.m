

clear all; close all; clc;
warning off all

load('../data/AR/ARwuzao.mat')
load('../data/AR/ARwuzao_idx_0.5train.mat')

num = 1;
lamda = zeros(1,2);   %需要调整的参数

lamda(1,2) = 0.001;   %miu参数
numHN=1400;

acc_test = zeros(num,9);
acc_train = zeros(num,9);
time = zeros(num,9);

num_idx=1;

h1 = 1e-7;
for j=11:15
    num_idx=j;
    lamda(1,1) = 1;
    for i =1:num 
        [acc_train(i,j),acc_test(i,j),time(i,j)] = main_L21_Multi_view(X,train_idx,test_idx,lamda,numHN,num_idx);
        %disp(sprintf('lamda_1:%g lamda_2:%d  i%d --  %6.4f',lamda(1,1),lamda(1,2),i,acc_test(i,j)));
        disp(sprintf('lamda_1:%g lamda_2:%g  --  %6.4f',lamda(1,1),lamda(1,2),acc_test(i,j)));
    end
    h1=h1*10;
end