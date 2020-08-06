function [acc_train,acc_test,time] = main_LR_Multi_view(X,trainIdx,testIdx,num_idx)
%{
****�����
****2017-11-22 ��
%}

 % ��num���������
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
time = 0;
fun=[];
for num_view = 1:v
    %disp(strcat('��',num2str(num_view),'��view'))
   
    TrnLabels = train_label';  % ��������ʾ��ǩ
    TestLabels = test_label';
    trainBase = train_data{1,num_view};
    testBase = test_data{1,num_view};
    
    X1=[trainBase testBase];   %ÿ�б�ʾһ������
    [D,N] = size(X1);
    normX1=zeros(D,N);
    for i=1:N                                                %��һ��
        normX1(:,i)=X1(:,i)/norm(X1(:,i));
    end
    X1 = normX1;
    clear normX1 D N;
    TrnData1 =X1(:,1:num_train);
    TestData1 = X1(:,num_train+1:end);
    
    TrnData = bsxfun(@minus, TrnData1, mean(TrnData1,2));     %ȥ���Ļ�
    TestData = bsxfun(@minus, TestData1, mean(TrnData1,2));

    TrainingData_File = [TrnLabels TrnData'];   %��װ����
    TestingData_File = [TestLabels TestData'];

    tic 
    [test_result,train_result] = LR(TrainingData_File, TestingData_File);
    tmp = toc;
    
    time = time + tmp;
    y_test{1,num_view} = test_result;
    y_train{1,num_view} = train_result;
end


%% ����׼ȷ��

%����������ݶ��Ӿ��ۼƸ���
s = zeros(num_test,c);   
for i = 1:v
    s = s + y_test{1,i};
end

y_test_total = zeros(1,num_test);
[~,idx] = max(s,[],2);
for j = 1:num_test
    y_test_total(1,j)=class(idx(j));
end

acc_test = (sum(test_label==y_test_total)/num_test)*100;

%����ѵ�����ݶ��Ӿ��ۼƸ���
t = zeros(num_train,c);   
for i = 1:v
    t = t + y_train{1,i};
end

y_train_total = zeros(1,num_train);
[~,idx] = max(t,[],2);
for j = 1:num_train
    y_train_total(1,j)=class(idx(j));
end

acc_train = (sum(train_label==y_train_total)/num_train)*100;
