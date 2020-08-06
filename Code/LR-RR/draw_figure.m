a(1,1) = 3.23/0.6;
a(1,2) = 6.74/3.03;
a(1,3) = 19.53/15.02;
a(1,4) = 30.77/27.19;
a(1,5) = 105.28/280.1;

b = [620,1160,1780,2560,4340];
plot(b,a)

fh = figure;


h = plot(b,a,'-ob',...
        'LineWidth',1);
grid on;
ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 1;  %设置网格线透明度
ax.LineWidth = 1;  %设置坐标轴框的线宽
ax.XLabel.String = 'Dimension of data';
ax.YLabel.String = 'Training time ratio ';