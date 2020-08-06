function [beta, A] = get_L21_elm_solve(H,Y,lambda,miu)
%{
�������������շ�����L21_ELM������Լ������beta=A
    ����betaʱ��ģ�ͱ�Ϊ��L21+F������ƽ�������ο����õġ�L21+F������ƽ�����ľ���ģ�ͽⷨ
    ����Aʱ��ģ�ͱ�Ϊ��F������ƽ��+F������ƽ�����������Ϊ�㼴�ɽ�
    ����Mʱ��M = M + miu*(beta-A);
%}
numRowBeta = size(H,2);
numColBeta = size(Y,2);
A = zeros(numRowBeta, numColBeta);
beta = zeros(numRowBeta, numColBeta);
M = zeros(numRowBeta, numColBeta);
I = ones(numRowBeta);
for iter = 1:50
    %����beta
    for i = 1:numColBeta
        cha = A-M/miu;
        normCha = norm(cha(:,i),2);
        if (lambda/miu) < normCha
            beta(:,i) = (normCha - (lambda/miu)) * cha(:,i) / normCha;
        else
            beta(:,i) = 0;
        end
    end
    %����A
    A = pinv(2*H'*H + miu*I) * (2*H'*Y + miu*beta + M);
    %����M
    M = M + miu*(beta-A);
    %����miu
    miu = 1.1 * miu;
    %����ÿ�ε�����Ŀ�꺯��
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