function [acc_train,acc_test,time] = main_NNLR_Multi_view(X,trainIdx,testIdx,lamda,num_idx)
%{
****武林璐
****2017-11-22 晚
****MSRC库
%}


 % 第num组随机索引
v = size(X,2)-1;
class = unique(X{1,end});
c = length(class);
test_label = X{1,end}(testIdx(num_idx,:));
train_label = X{1,end}(trainIdx(num_idx,:));
test_data = cell(1,v);
train_data = cell(1,v);
for j = 1:v
    test_data{1,j} = X{1,j}(:,testIdx(num_idx,:));
    train_data{1,j} = X{1,j}(:,trainIdx(num_idx,:));
end
num_train = size(train_data{1,1},2);
num_test = size(test_data{1,1},2);


%NumberofHiddenNeurons = 600;
res=[];
accuracy = [];
y_test = cell(1,v);
y_train = cell(1,v);
time = 0;
fun=[];
for num_view = 1:v
    %disp(strcat('第',num2str(num_view),'个view'))
   
    TrnLabels = train_label';  % 列向量表示标签
    TestLabels = test_label';
    trainBase = train_data{1,num_view};
    testBase = test_data{1,num_view};
    
    X1=[trainBase testBase];   %每列表示一个数据
    [D,N] = size(X1);
    normX1=zeros(D,N);
    for i=1:N                                                %归一化
        normX1(:,i)=X1(:,i)/norm(X1(:,i));
        %           normX1(:,i)=X1(:,i)/1;
    end
    X1 = normX1;
    clear normX1 D N;
    TrnData1 =X1(:,1:num_train);
    TestData1 = X1(:,num_train+1:end);
    %分块处理
    %          Trn = reshape(TrnData1,[size(TrnData1,2),ImgSize1,ImgSize2]);
    %          im = im2col_general(InImg{i},[PatchSize PatchSize]);
    
    TrnData = bsxfun(@minus, TrnData1, mean(TrnData1,2));     %去中心化
    TestData = bsxfun(@minus, TestData1, mean(TrnData1,2));
    TrainingData_File = [TrnLabels TrnData'];   %组装数据
    TestingData_File = [TestLabels TestData'];
    Elm_Type =1;
    NumberofHiddenNeurons=700;
    ActivationFunction ='sig';
    %% 单独一层ELM
    tic
    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = LRR_ELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction);
    tmp = toc;
    time = time + tmp;
end
acc_train = TrainingAccuracy;
acc_test = TestingAccuracy; 
