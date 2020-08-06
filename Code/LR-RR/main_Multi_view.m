function [acc_train,acc_test,time] = main_Multi_view(X,trainIdx,testIdx,lamda,num_idx)
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


res=[];
accuracy = [];
y_test = cell(1,v);
y_train = cell(1,v);
total_time = 0;
fun=[];
for num_view = 1:v
    %disp(strcat('第',num2str(num_view),'个view'))
    trainBase = train_data{1,num_view};
    testBase = test_data{1,num_view};
    
    X1=[trainBase testBase];   %每列表示一个数据
    [D,N] = size(X1);
    normX1=zeros(D,N);
    for i=1:N                                                %归一化
        normX1(:,i)=X1(:,i)/norm(X1(:,i));
    end
    X1 = normX1;
    clear normX1 D N;
    TrnData1 =X1(:,1:num_train);
    TestData1 = X1(:,num_train+1:end);

    TrnData = bsxfun(@minus, TrnData1, mean(TrnData1,2));     %去中心化
    TestData = bsxfun(@minus, TestData1, mean(TrnData1,2));

    [Beta,time] = LR_RR(TrnData,train_label);
    
    total_time = total_time + time;
    y_test{1,num_view} = TestData' * Beta;
    y_train{1,num_view} = TrnData' * Beta;
end

s = zeros(num_test,c);   %计算测试误差
for i = 1:v
    s = s + y_test{1,i};
end
y_test_total = zeros(1,num_test);
[~,idx] = max(s,[],2);
for j = 1:num_test
    y_test_total(1,j)=class(idx(j));
end

acc_test = (sum(test_label==y_test_total)/num_test)*100;

t = zeros(num_train,c);   %计算训练误差
for i = 1:v
    t = t + y_train{1,i};
end

y_train_total = zeros(1,num_train);
[~,idx] = max(t,[],2);
for j = 1:num_train
    y_train_total(1,j)=class(idx(j));
end

acc_train = (sum(train_label==y_train_total)/num_train)*100;

