function [acc_train,acc_test,time] = H_D_LR(train_data,train_label,test_data,test_label,lamda,s,multi_view_predict,num_HN)
%  MLR ; Multi-view low-rank regression
%  H_D_LR : 隐藏单元输出的多视觉低秩分解
%  Input: 
%        train_data(1,v): train_data matrix; 
%                cell数据类型，包含v个矩阵,分别表示v个view
%                每个view（p_v,n）中，每一列表示一个样本数据，共包含n个样本，每个view包含p_v个特征
%        train_label(1,n): class indicator matrix; 
%               行向量，表示训练数据的标签
%        lamda(1,v): 正则权重参数
%        s: 子空间维度，s<min(p_v,c)
%  output:
%        A(1,v); cell数据类型,其中每个元素A_v的size为(p_v,s)
%        B(s,c)

%% 计算全局变量
v = size(train_data,2);
p = zeros(1,v);
num_train = size(train_data{1,1},2);
num_test = size(test_data{1,1},2);


for i = 1:v
    p(1,i) = size(train_data{1,i},1);
end

train_label_origin = train_label;
class = unique(train_label);
c = length(class);
train_label = one_hot_encode(train_label)';

%{
X = zeros(sum(p),n);
index = 1;
for i = 1:v
    X(index:index+p(1,i)-1,:) = train_data{1,i};
    index = index+p(1,i);
end
%}
%%   归一化和去中心化

for i= 1:v     %训练和测试数据归一化和去中心化

    % 按样本进行2范数归一化，然后去中心化
    tmp_train = train_data{1,i};       %训练数据按样本（列）归一化
    for j = 1:num_train
        tmp_train(:,j) = tmp_train(:,j)/norm(tmp_train(:,j));
    end
    
    tmp_test = test_data{1,i};        %测试数据按样本归（列）一化
    for j = 1:num_test
        tmp_test(:,j) = tmp_test(:,j)/norm(tmp_test(:,j));
    end
    
    %train_data{1,i} = tmp_train*2-1;
    %test_data{1,i} = tmp_test*2-1;
    train_data{1,i} = tmp_train - repmat(mean(tmp_train,2),[1,size(tmp_train,2)]); %训练数据去中心化 
    test_data{1,i} = tmp_test - repmat(mean(tmp_train,2),[1,size(tmp_test,2)]); %测试数据去中心化
    
    %{
    % 参照ELM输入数据的预处理
    % 在特征维度上，使用最大-最小归一化方式将特征归一化到[0,1]之间，进而变换到[-1,1] 区间
    tmp_data = [train_data{1,i},test_data{1,i}];
    for j = 1:p(1,i)
        tmp_data(j,:) = (tmp_data(j,:) - min(tmp_data(j,:))/(max(tmp_data(j,:))-min(tmp_data(j,:))));
    end
    tmp_train= tmp_data(:,1:num_train);
    tmp_test = tmp_data(:,num_train+1:num_train+num_test);
    
    train_data{1,i} = tmp_train - repmat(mean(tmp_train,2),[1,size(tmp_train,2)]); %训练数据去中心化 
    test_data{1,i} = tmp_test - repmat(mean(tmp_train,2),[1,size(tmp_test,2)]); %测试数据去中心化
    %}
end


train_label_norm = max(1e-14,full(sum(train_label.^2,1)));     %训练数据标签(train_label)按类别（列）归一化
train_label_norm = train_label_norm.^-.5;
train_label = train_label./train_label_norm(ones(num_train,1),:);

train_label_mean = mean(train_label,1);            
train_label = train_label - repmat(train_label_mean,[num_train,1]);    %去中心化train_label

%% 计算lamda = p90
%{
for i = 1:v
    tmp = train_data{1,i}*train_data{1,i}';
    [~,value] = eig(tmp);
    value = diag(value);
    p90 = floor(sum(value>1e-14)*0.9);
    [~,idx] = sort(value,1,'descend');   
    lamda(1,i) = value(idx(p90));
end
%}

tic
%% 训练和测试数据进行随机投影 
%rate = 1;
%num_HN = floor(p*rate);
%num_HN = ones(1,v)*200;
p = num_HN;   % 更新p(1,v)

input_weigth_base = rand(10000,10000)*2-1;  %随机生成网络结构
bias_HN_base = rand(10000,1);

H_train = random_projection(train_data,num_HN,input_weigth_base,bias_HN_base);
H_test = random_projection(test_data,num_HN,input_weigth_base,bias_HN_base);

for i = 1:v
    tmp_H_train = H_train{1,i};     % 训练数据归一化
    %for j = 1:num_train
    %    tmp_H_train(:,j) = tmp_H_train(:,j)./norm(tmp_H_train(:,j));
    %end
    
    tmp_H_test = H_test{1,i};   % 测试数据归一化
    %for j = 1:num_test
    %    tmp_H_test(:,j) = tmp_H_test(:,j)./norm(tmp_H_test(:,j));
    %end
    
    
    H_train{1,i} = tmp_H_train - repmat(mean(tmp_H_train,2),[1,num_train]);%训练数据去中心化
    
    H_test{1,i} = tmp_H_test - repmat(mean(tmp_H_train,2),[1,num_test]);%测试数据去中心化
end


%更新数据
train_data = H_train;
test_data = H_test;

%% compute S_b和S_t
X = zeros(sum(p),num_test);
index = 1;
for i = 1:v
    X(index:index+p(1,i)-1,:) = train_data{1,i};
    index = index+p(1,i);
end

S_b = X*train_label*train_label'*X';

S_t = zeros(sum(p));
index=1;
for i = 1:v
    x = train_data{1,i};
    S_t(index:index+p(1,i)-1,index:index+p(1,i)-1) = x*x'+ lamda(1,i)*eye(p(1,i));
    index = index+p(1,i);
end
%% compute A

[eigVector,eigValue] = eig(S_t\S_b);
eigValue = diag(eigValue);
[~,index] = sort(eigValue,1,'descend');
A_tmp = eigVector(:,index(1:s));

A = cell(1,v);           %将各个视觉的投影矩阵分离开
idx = 1;
for i = 1:v
    A{1,i} = A_tmp(idx:idx+p(1,i)-1,:);
    idx = idx+p(1,i);
end

%% compute B
G = zeros(s,s);
H = zeros(s,c);
for i = 1:v
    G = G+A{1,i}'*(train_data{1,i}*train_data{1,i}'+lamda(1,i)*eye(p(1,i)))*A{1,i};
    H = H+A{1,i}'*train_data{1,i}*train_label;
end
B = inv(G)*H;

%% calculate accurractrain_label

y = zeros(1,num_test);
switch lower(multi_view_predict)
    case {'sum'}
        %先对各视觉的预测求和，然后取最大值作为最终的预测结果
        s = zeros(num_test,c);
        for i = 1:v
            tmp = test_data{1,i}'*A{1,i}*B;
            s = s+tmp;
        end
        
        [~,idx] = max(s,[],2);
        for j = 1:num_test
            y(1,j)=class(idx(j));
        end
    case {'voting'}
        s = zeros(num_test,c);
        for i = 1:v
            tmp = test_data{1,i}'*A{1,i}*B;
            [~,idx] = max(tmp,[],2);
            for j = 1:num_test
                s(j,idx(j))=1;
            end
        end
        [~,idx] = max(s,[],2);
        for i = 1:num_test
            y(1,i) = class(idx(i));
        end
end

acc_test = (sum(test_label==y)/num_test)*100;

%计算训练误差
y_train = zeros(1,num_train);
t = zeros(num_train,c);
for i = 1:v
    tmp = train_data{1,i}'*A{1,i}*B;
    t = t+tmp;
end

[~,idx_train] = max(t,[],2);
for j = 1:num_train
    y_train(1,j)=class(idx_train(j));
end
acc_train = (sum(train_label_origin==y_train)/num_train)*100;
time = toc;
