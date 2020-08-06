function res = one_hot_encode(Y)
% Y 行向量，表示样本的类别
% res(c,n) Y向量的one-hot编码

class = unique(Y);
c = length(class);
n = size(Y,2);
res = zeros(c,n);

for i = 1:n
    for j = 1:c
        if Y(1,i) == class(1,j)
            break;
        end
    end
    res(j,i) = 1;
end