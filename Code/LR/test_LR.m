clear all; close all; clc;
warning off all

%load('../data/BinAlpha36/BinAlpha36.mat')
%load('../data/BinAlpha36/BinAlpha36_idx_20train.mat')
%load('../data/usps/usps.mat')
%load('../data/usps/usps_idx_0.5train.mat')
%load('../data/umist/umist.mat')
%load('../data/umist/umist_idx_0.5train.mat')
%load('../data/COIL20/coil20.mat')
%load('../data/COIL20/coil20_idx_0.5train.mat')
%load('../data/AR/ARwuzao.mat')
%load('../data/AR/ARwuzao_idx_0.5train.mat')
load('../data/Cal101/Cal101_HOG&LBP.mat')
load('../data/Cal101/Cal101_idx_40train.mat')
%load('../data/MSRC/MSRC.mat')
%load('../data/MSRC/MSRC_idx_0.5train.mat')

num = 1;
acc_test = zeros(num,9);
acc_train = zeros(num,9);
time = zeros(num,9);
num_idx=1;

for j=1:15
    num_idx=j;
    for i =1:num
        [acc_train(i,j),acc_test(i,j),time(i,j)] = main_LR_Multi_view(X,train_idx,test_idx,num_idx);
        disp(sprintf('num:%g  --  %6.4f',num_idx,acc_test(i,j)));
    end
end