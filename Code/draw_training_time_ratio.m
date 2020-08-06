c = [1.77,2.6,6.77,13.48,6.63];
b = [0.59,2.96,15.6,27.35,282];
x = [620,1160,1780,2560,4340];
 plot(x,c./b,'o-','LineWidth',2);
hold on
plot(500:1:4500,ones(1,length(500:1:4500)),'--')
grid on
xlabel('r'),ylabel('dimension of datasets') 
ylabel('r'),xlabel('dimension of datasets')
ylabel('training time ratio'),xlabel('dimension of datasets')
legend('r','r = 1')
