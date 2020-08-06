function y = get_L21_norm(A,flag)
%{
flag = 1:对每列取2范数，再相加
flag = 2:对每行取2范数，再相加
%}
[numRow,numCol] = size(A);

if flag == 1
    tmp = zeros(1,numCol);
    for i=1:numCol
        tmp(i) = norm(A(:,i),2);
    end  
elseif flag == 2
    tmp = zeros(1,numRow);
    for i=1:numRow
        tmp(i) = norm(A(i,:),2);
    end  
end
y = sum(tmp);