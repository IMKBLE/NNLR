

clear all; close all; clc;
warning off all

load('../data/Cal101/Cal101_HOG&LBP.mat')
load('../data/Cal101/Cal101_idx_40train.mat')

num = 1;
acc_test = zeros(num,15);
acc_train = zeros(num,15);
time = zeros(num,15);

lamda=0.01;
num_idx=1;
h=1e-7;

for j=1:15
    num_idx=j;
    lamda = 0.1;
    for i =1:num
        [acc_train(i,j),acc_test(i,j),time(i,j)] = main_RR_Multi_view(X,train_idx,test_idx,num_idx,lamda);
        %disp(sprintf('lamda_1:%g lamda_2:%d  i%d --  %6.4f',lamda(1,1),lamda(1,2),i,acc_test(i,j)));
        disp(sprintf('num:%g lamda:%g --  %6.4f',num_idx,lamda,acc_test(i,j)));
    end
    h=h*10;
end
