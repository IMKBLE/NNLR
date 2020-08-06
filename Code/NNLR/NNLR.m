function [Beta, time] = NNLR(train_data,train_label,lamda)


Y = one_hot_encode(train_label)*2-1;
X = train_data;   

lamda_1 = lamda(1,1);
lamda_2 = lamda(1,2);
u=0.1; 
u_max=1e+10; 
alpha = 1.3;    % 表示u的增长速度
fun = [];

tic
% 初始化
Beta = zeros(size(X,1),size(Y,1)); % 系数矩阵
M = zeros(size(Y)); 
A = zeros(size(Y));
for i=1:100
  %% 固定权重,更新M
   C = Beta' * X +A./u;
   [U,S,V] = svd(C);
   r = rank(C);
   S = S(:,1:r);
   D = diag(diag(S)-lamda_2/u);%  lamda_2/u?
   D(D<0)=0;
   M = U(:,1:r)*D*V(:,1:r)';
  %% 固定M,更新权重
   Beta = ((2+u)*X*X'+2*lamda_1*eye(size(X*X'))) \ (u*X*M' + 2*X*Y'- X*A');
   fval = norm(Beta' * X-M,'fro').^2;
   %[~,s,~] = svd(M);
   %fval = norm(OutputWeight' * H-T,'fro').^2 + lamda_1*(norm(OutputWeight,'fro').^2) + lamda_2*sum(diag(s))+(u/2)*(norm(A./u + M - OutputWeight' * H).^2);
   fun = [fun fval];
   if fval<1e-7
       break;
   end  
   A = A + u*(Beta' * X-M);
   u = min(alpha*u,u_max);
end
save('fun.mat','fun');
time = toc;

