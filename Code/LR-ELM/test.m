
clear all; close all; clc;
warning off all

load('../data/Cal101/Cal101_sift.mat')
load('../data/Cal101/Cal101_idx_40train.mat')
%load('../data/BinAlpha36/BinAlpha36.mat')
%load('../data/BinAlpha36/BinAlpha36_idx_20train.mat')
load('../data/usps/usps.mat')
load('../data/usps/usps_idx_0.5train.mat')
%load('../data/umist/umist.mat')
%load('../data/umist/umist_idx_0.5train.mat')
%load('../data/COIL20/coil20.mat')
%load('../data/COIL20/coil20_idx_0.5train.mat')
%load('../data/AR/ARwuzao.mat')
%load('../data/AR/ARwuzao_idx_0.5train.mat')
%load('../data/YaleB/YaleB.mat')
%load('../data/YaleB/YaleB_idx_0.5train.mat')
num = 1;
lamda = zeros(1,2);   %需要调整的参数
lamda(1,2) = 0.1;
lamda(1,1) = 0.4;

acc_test = zeros(num,9);
acc_train = zeros(num,9);
time = zeros(num,9);
num_idx=1;
res=[];

for j=1:15
    num_idx = j;
    lamda(1,2) = 5;
    lamda(1,1) = 0.5;
    [acc_train(1,j),acc_test(1,j),time(1,j)] = main_Multi_view(X,train_idx,test_idx,lamda,num_idx);
    disp(sprintf('lamda_1:%g lamda_2:%g  --  %6.4f  %6.4f',lamda(1,1),lamda(1,2),acc_test(1,j),acc_train(1,j)));

end
