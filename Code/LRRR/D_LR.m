function [acc_train,acc_test,time] = D_LR(train_data,train_label,test_data,test_label,lamda,s,multi_view_predict)
%  MLR ; Multi-view low-rank regression
%  D_LR : low-rank decomposition
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

    train_data{1,i} = tmp_train - repmat(mean(tmp_train,2),[1,size(tmp_train,2)]); %训练数据去中心化 
    test_data{1,i} = tmp_test - repmat(mean(tmp_train,2),[1,size(tmp_test,2)]); %测试数据去中心化

end


train_label_norm = max(1e-14,full(sum(train_label.^2,1)));     %训练数据标签(train_label)按类别（列）归一化
train_label_norm = train_label_norm.^-.5;
train_label = train_label./train_label_norm(ones(num_train,1),:);

train_label_mean = mean(train_label,1);            
train_label = train_label - repmat(train_label_mean,[num_train,1]);    %去中心化train_label


tic
%% compute S_b和S_t
X = zeros(sum(p),num_train);
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
m = A{1,1}*B;
save('../DLR_coefficient_matrix.mat','m');
%% calculate accurractrain_label

y_test = zeros(1,num_test);
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
            y_test(1,j)=class(idx(j));
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
            y_test(1,i) = class(idx(i));
        end
end

acc_test = (sum(test_label==y_test)/num_test)*100;

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
