load('../data/YaleB/YaleB.mat')
load('../data/YaleB/YaleB_idx_0.5train.mat')

label = X{1,2};
train_label = label(train_idx(1,:));

res = one_hot_encode(train_label);
res = res';
[U,S,V] = svd(res);