function [acc_train,acc_test,time] = H_D_LR(train_data,train_label,test_data,test_label,lamda,s,multi_view_predict,num_HN)
%  MLR ; Multi-view low-rank regression
%  H_D_LR : ���ص�Ԫ����Ķ��Ӿ����ȷֽ�
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

%{
X = zeros(sum(p),n);
index = 1;
for i = 1:v
    X(index:index+p(1,i)-1,:) = train_data{1,i};
    index = index+p(1,i);
end
%}
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
    
    %train_data{1,i} = tmp_train*2-1;
    %test_data{1,i} = tmp_test*2-1;
    train_data{1,i} = tmp_train - repmat(mean(tmp_train,2),[1,size(tmp_train,2)]); %ѵ������ȥ���Ļ� 
    test_data{1,i} = tmp_test - repmat(mean(tmp_train,2),[1,size(tmp_test,2)]); %��������ȥ���Ļ�
    
    %{
    % ����ELM�������ݵ�Ԥ����
    % ������ά���ϣ�ʹ�����-��С��һ����ʽ��������һ����[0,1]֮�䣬�����任��[-1,1] ����
    tmp_data = [train_data{1,i},test_data{1,i}];
    for j = 1:p(1,i)
        tmp_data(j,:) = (tmp_data(j,:) - min(tmp_data(j,:))/(max(tmp_data(j,:))-min(tmp_data(j,:))));
    end
    tmp_train= tmp_data(:,1:num_train);
    tmp_test = tmp_data(:,num_train+1:num_train+num_test);
    
    train_data{1,i} = tmp_train - repmat(mean(tmp_train,2),[1,size(tmp_train,2)]); %ѵ������ȥ���Ļ� 
    test_data{1,i} = tmp_test - repmat(mean(tmp_train,2),[1,size(tmp_test,2)]); %��������ȥ���Ļ�
    %}
end


train_label_norm = max(1e-14,full(sum(train_label.^2,1)));     %ѵ�����ݱ�ǩ(train_label)������У���һ��
train_label_norm = train_label_norm.^-.5;
train_label = train_label./train_label_norm(ones(num_train,1),:);

train_label_mean = mean(train_label,1);            
train_label = train_label - repmat(train_label_mean,[num_train,1]);    %ȥ���Ļ�train_label

%% ����lamda = p90
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
%% ѵ���Ͳ������ݽ������ͶӰ 
%rate = 1;
%num_HN = floor(p*rate);
%num_HN = ones(1,v)*200;
p = num_HN;   % ����p(1,v)

input_weigth_base = rand(10000,10000)*2-1;  %�����������ṹ
bias_HN_base = rand(10000,1);

H_train = random_projection(train_data,num_HN,input_weigth_base,bias_HN_base);
H_test = random_projection(test_data,num_HN,input_weigth_base,bias_HN_base);

for i = 1:v
    tmp_H_train = H_train{1,i};     % ѵ�����ݹ�һ��
    %for j = 1:num_train
    %    tmp_H_train(:,j) = tmp_H_train(:,j)./norm(tmp_H_train(:,j));
    %end
    
    tmp_H_test = H_test{1,i};   % �������ݹ�һ��
    %for j = 1:num_test
    %    tmp_H_test(:,j) = tmp_H_test(:,j)./norm(tmp_H_test(:,j));
    %end
    
    
    H_train{1,i} = tmp_H_train - repmat(mean(tmp_H_train,2),[1,num_train]);%ѵ������ȥ���Ļ�
    
    H_test{1,i} = tmp_H_test - repmat(mean(tmp_H_train,2),[1,num_test]);%��������ȥ���Ļ�
end


%��������
train_data = H_train;
test_data = H_test;

%% compute S_b��S_t
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

%% calculate accurractrain_label

y = zeros(1,num_test);
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
