function [acc_train,acc_test,time] = D_LR(train_data,train_label,test_data,test_label,lamda,s,multi_view_predict)
%  MLR ; Multi-view low-rank regression
%  D_LR : low-rank decomposition
%  Input: 
%        train_data(1,v): train_data matrix; 
%                cell�������ͣ�����v������,�ֱ��ʾv��view
%                ÿ��view��p_v,n���У�ÿһ�б�ʾһ���������ݣ�������n��������ÿ��view����p_v������
%        train_label(1,n): class indicator matrix; 
%               ����������ʾѵ�����ݵı�ǩ
%        lamda(1,v): ����Ȩ�ز���
%        s: �ӿռ�ά�ȣ�s<min(p_v,c)
%  output:
%        A(1,v); cell��������,����ÿ��Ԫ��A_v��sizeΪ(p_v,s)
%        B(s,c)

%% ����ȫ�ֱ���
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

%%   ��һ����ȥ���Ļ�

for i= 1:v     %ѵ���Ͳ������ݹ�һ����ȥ���Ļ�

    % ����������2������һ����Ȼ��ȥ���Ļ�
    tmp_train = train_data{1,i};       %ѵ�����ݰ��������У���һ��
    for j = 1:num_train
        tmp_train(:,j) = tmp_train(:,j)/norm(tmp_train(:,j));
    end
    
    tmp_test = test_data{1,i};        %�������ݰ������飨�У�һ��
    for j = 1:num_test
        tmp_test(:,j) = tmp_test(:,j)/norm(tmp_test(:,j));
    end

    train_data{1,i} = tmp_train - repmat(mean(tmp_train,2),[1,size(tmp_train,2)]); %ѵ������ȥ���Ļ� 
    test_data{1,i} = tmp_test - repmat(mean(tmp_train,2),[1,size(tmp_test,2)]); %��������ȥ���Ļ�

end


train_label_norm = max(1e-14,full(sum(train_label.^2,1)));     %ѵ�����ݱ�ǩ(train_label)������У���һ��
train_label_norm = train_label_norm.^-.5;
train_label = train_label./train_label_norm(ones(num_train,1),:);

train_label_mean = mean(train_label,1);            
train_label = train_label - repmat(train_label_mean,[num_train,1]);    %ȥ���Ļ�train_label


tic
%% compute S_b��S_t
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

A = cell(1,v);           %�������Ӿ���ͶӰ������뿪
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
        %�ȶԸ��Ӿ���Ԥ����ͣ�Ȼ��ȡ���ֵ��Ϊ���յ�Ԥ����
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

%����ѵ�����
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
