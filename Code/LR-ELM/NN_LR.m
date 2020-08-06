function [TY,Y] =NN_LR(TrainingData_File, TestingData_File, Elm_Type,lamda)

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);%这是原来程序，要求输入训练数据文件路径
train_data = TrainingData_File;%% 这里TrainingData_File代表的是训练数据数据
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';              %   每列代表一个样本
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
% test_data=load(TestingData_File);%这是原来程序，要求输入测试数据文件路径
test_data=TestingData_File;
TV.T=test_data(:,1)';%% 这里TrainingData_File代表的是训练数据数据
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);            %   Concatenate arrays.
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;                                 % 统计类别个数和标签
    NumberofOutputNeurons=number_class;
       
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;   %？？？0->-1;1->1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
H = P;
OutputWeight = zeros(size(H,1),size(T,1));
M = zeros(size(T));
A = zeros(size(T));
%lamda_1=0.1; %最初赵双双设的是0.1
% lamda_2=0.1; % lamda_2
lamda_1 = lamda(1,1);
lamda_2 = lamda(1,2);
u=0.1; 
u_max=1e+10; 
%alpha = 1.3;    % 表示u的增长速度
alpha=1.3;
fun = [];
for i=1:100
  %% 固定权重,更新M
   C = OutputWeight' * H +A./u;
   [U,S,V] = svd(C);
   r = rank(C);
   S = S(:,1:r);
   D = diag(diag(S)-lamda_2/u);%  lamda_2/u?
   D(D<0)=0;
   M = U(:,1:r)*D*V(:,1:r)';
  %% 固定M,更新权重
   OutputWeight = ((2+u)*H*H'+2*lamda_1*eye(size(H*H'))) \ (u*H*M' + 2*H*T'- H*A');
   fval = norm(OutputWeight' * H-M,'fro').^2;
   %[~,s,~] = svd(M);
   %fval = norm(OutputWeight' * H-T,'fro').^2 + lamda_1*(norm(OutputWeight,'fro').^2) + lamda_2*sum(diag(s))+(u/2)*(norm(A./u + M - OutputWeight' * H).^2);
   fun = [fun fval];
   if fval<1e-20
       break;
   end  
   A = A + u*(OutputWeight' * H-M);
   u = min(alpha*u,u_max);
end
end_time_train=cputime;
%TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM

save('../NNLR_coefficient_matrix.mat','OutputWeight');
%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
H_test = TV.P; % 去掉随机投影
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
%end_time_test=cputime;
%TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;
end