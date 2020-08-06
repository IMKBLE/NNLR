function [T, time] = LR_RR(train_data,train_label)


Y = one_hot_encode(train_label)*2-1;
X = train_data;

class = unique(train_label);
c = length(class);  % 类别个数
n = size(train_label, 2);  % 样本数量
d = size(train_data, 1);

%参数
eta = 1;  % F范数项系数
lambda = 0.1;  % 1范数项的收缩系数
rho = 1.2;  
gamma = 1.1;  % 正则项系数
W = diag(ones(1, c));

%初始化
tic
D = X;
D_ = [X;ones(1, n)];
E = X - D;
T = pinv(D_*D_'+gamma*eye(d+1))*D_*Y';
F1 = X/norm(X);
F2 = D_/norm(D_);
mu1 = ((d*n)/4)*norm(X, 1);
mu2 = ((d*n)/4)*norm(D_, 1);
%求解
while((norm(X-D-E, 'fro')/(norm(X, 'fro')) < 1e-8) && ((norm(D_ - [D;ones(1, n)], 'fro')/norm(D_, 'fro')) < 1e-8))
%for iii=1:10
    T = pinv(D_*D_'+gamma*eye(d+1))*D_*Y';
    D_ = pinv(eta*T*W'*W*T' + mu2*eye(d+1))*(eta*T*W'*Y - F2 + mu2*[D; ones(1, n)]);
    tmp = F2 + mu2*D_;
    beta = mu1+mu2;
    Z = (1/beta)*(F1 + mu1*(X-E) + tmp(1:d,:));
    [U, S, V] = svd(Z);
    r = rank(Z);
    S = S(:,1:r);
    S = diag(diag(S) - 1/beta);
    S(S<0) = 0;
    D = U(:,1:r)*S*V(:,1:r)';
    tmp = X-D+F1*(1/mu1);
    t = (lambda/mu1);
    for i=1:d
        for j=1:n
            if tmp(i,j)>t
                E(i,j) = tmp(i,j)-t;
            elseif tmp(i,j) < -t
                E(i,j) = tmp(i,j)+t;
            else
                E(i,j) = 0;
            end
        end
    end
    F1 = F1 + mu1*(X-D-E);
    F2 = F2 + mu2*(D_-[D;ones(1,n)]);
    mu1 = rho*mu1;
    mu2 = rho*mu2;
end
time = toc;
T = T(1:d,:);

    

