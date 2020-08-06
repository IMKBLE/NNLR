
clear all; close all; clc;
addpath(genpath(pwd));

load('../data/Cal101/Cal101_HOG&LBP.mat')
load('../data/Cal101/Cal101_idx_40train.mat')


v = size(X,2)-1;
train_data = cell(1,v);
test_data = cell(1,v);

%% 设置参数
multi_view_predict = 'sum';

lamda = ones(1,v)*0.15;
s = 9;

num = 1;      
%%
acc_test = zeros(num,6);
acc_train = zeros(num,6);
time = zeros(num,6);

p = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2];

num_idx=1;
h = 1e-7;
%% 
for j = 1:15
    for i = 1:v
        data = X{1,i};
        train_data{1,i} = data(:,train_idx(num_idx,:));
        test_data{1,i} = data(:,test_idx(num_idx,:));
    end
    label = X{1,v+1};
    train_label = label(train_idx(num_idx,:));
    test_label = label(test_idx(num_idx,:));
    
        num_idx=j;  %表示第num_idx组随机索引
        s = 6;
        lamda = ones(1,v)*1;
        [acc_train(1,j),acc_test(1,j),time(1,j)] = D_LR(train_data,train_label,test_data,test_label,lamda,s,multi_view_predict);
        %[acc_train(i,j),acc_test(i,j),time(i,j)] = H_D_LR(train_data,train_label,test_data,test_label,lamda,s,multi_view_predict,numHN);
        %disp(sprintf('s:%d  i%d --  %6.4f',s,i,acc(i,j)));
        disp(sprintf('lamda:%g s:%g i%d --  %6.4f  --%6.4f',lamda(1,1),s,i,acc_test(1,j),acc_train(1,j)));
         h = h*10;
  
end