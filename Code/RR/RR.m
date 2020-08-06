
function [test_result,train_result] =RR(TrainingData_File, TestingData_File,lamda)

% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set

% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification


train_label=TrainingData_File(:,1)';
train_data=TrainingData_File(:,2:size(TrainingData_File,2))';       %train_data每列表示一个样本      
clear TrainingData_File;                                  

test_label=TestingData_File(:,1)';
test_data=TestingData_File(:,2:size(TestingData_File,2))';
clear TestingData_File;                                   

Y = one_hot_encode(train_label);
tmp_p = size(train_data,1);
w = pinv(train_data*train_data'+lamda*eye(tmp_p))*train_data*Y';
save('../RR_coefficient_matrix.mat','w');

test_result = test_data'*w;
train_result = train_data'*w;
