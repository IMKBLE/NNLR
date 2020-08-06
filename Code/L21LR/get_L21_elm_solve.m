function [beta, A] = get_L21_elm_solve(H,Y,lambda,miu)
%{
用增广拉格朗日方法解L21_ELM，构造约束条件beta=A
    更新beta时，模型变为【L21+F范数的平方】，参考张婷的【L21+F范数的平方】的经典模型解法
    更新A时，模型变为【F范数的平方+F范数的平方】，求导令导数为零即可解
    更新M时，M = M + miu*(beta-A);
%}
numRowBeta = size(H,2);
numColBeta = size(Y,2);
A = zeros(numRowBeta, numColBeta);
beta = zeros(numRowBeta, numColBeta);
M = zeros(numRowBeta, numColBeta);
I = ones(numRowBeta);
for iter = 1:50
    %更新beta
    for i = 1:numColBeta
        cha = A-M/miu;
        normCha = norm(cha(:,i),2);
        if (lambda/miu) < normCha
            beta(:,i) = (normCha - (lambda/miu)) * cha(:,i) / normCha;
        else
            beta(:,i) = 0;
        end
    end
    %更新A
    A = pinv(2*H'*H + miu*I) * (2*H'*Y + miu*beta + M);
    %更新M
    M = M + miu*(beta-A);
    %更新miu
    miu = 1.1 * miu;
    %计算每次迭代的目标函数
    obj(iter) = norm(H*beta-Y,'fro')^2 + lambda*get_L21_norm(beta,1);% + (miu/2)*norm(H*beta-A+M/miu,'fro')^2;
%     stopFlag = norm(beta-A,'fro')/norm(beta,'fro');
%     if stopFlag < 1e-7
%         break;
%     end
    if iter > 1 && (obj(iter)-obj(iter-1)) > 0
        break;
    end
end
% plot(obj)