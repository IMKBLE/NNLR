function H = random_projection(train_data,numHN,input_weigth_base,bias_HN_base)
%Input: 
%     train_data(1,v): train_data matrix; 
%                cell�������ͣ�����v������,�ֱ��ʾv��view
%                ÿ��view��p_v,n���У�ÿһ�б�ʾһ���������ݣ�������n��������ÿ��view����p_v������
%     numHN(1,v): Number of hidden neurons
%
v = size(train_data,2);
p = zeros(1,v);
for i = 1:v
    p(1,i) = size(train_data{1,i},1);
end
num_train = size(train_data{1,1},2);

H = cell(1,v);

for i = 1:v
    %input_weigth = rand(numHN(i),p(1,i))*2-1;
    %bias_HN = rand(numHN(i),1);
    input_weigth = input_weigth_base(1:numHN(i),1:p(1,i));  %ѡ������ṹ
    bias_HN = bias_HN_base(1:numHN(i),1);
    
    tmp_H = input_weigth * train_data{1,i};   
    bias_matrix = bias_HN(:,ones(1,num_train));
    tmp_H = tmp_H + bias_matrix;
    
    H{1,i} = 1./(1 + exp(-tmp_H));   % sigmoid�����
end