
clear all; close all; clc;
warning off all

load('F:\工作\NNLR小论文\程序及数据\data\MSRC\MSRC.mat')
load('F:\工作\NNLR小论文\程序及数据\data\MSRC\MSRC_idx_0.5train.mat')

num = 1;
lamda = zeros(1,2);   %需要调整的参数
lamda(1,2) = 0.1;
lamda(1,1) = 0.4;

acc_test = zeros(num,9);
acc_train = zeros(num,9);
time = zeros(num,9);
num_idx=1;

h = 1e-7;
for j=1:15
    num_idx = j;
    lamda(1,2) = 1;
    lamda(1,1) = 0.1;
    [acc_train(1,j),acc_test(1,j),time(1,j)] = main_NNLR_Multi_view(X,train_idx,test_idx,lamda,num_idx);
    %disp(sprintf('lamda_1:%g lamda_2:%d  i%d --  %6.4f',lamda(1,1),lamda(1,2),i,acc_test(i,j)));
    disp(sprintf('lamda_1:%g lamda_2:%g  --  %6.4f  %6.4f',lamda(1,1),lamda(1,2),acc_test(1,j),acc_train(1,j)));
    h=h*10;
end