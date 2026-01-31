%%测试平台区域
%% =========================================
% 方波傅里叶逼近（0~pi=1, -pi~0=-1）
% N-频率-平均高度误差分析
% MATLAB 2019a
% =========================================
clear; clc; close all;
% 设置中文字体
set(0, 'DefaultAxesFontName', 'SimHei');  % 设置默认坐标轴字体为黑体
set(0, 'DefaultTextFontName', 'SimHei');   % 设置默认文本字体为黑体
%% 1. 时间轴
t = linspace(-pi, pi, 8000);

%% 2. 理想方波（按你的定义）
f_ideal = -ones(size(t));
f_ideal(t > 0 & t < pi) = 1;

%% 3. 参数
N_list = [1 3 5 10 20  27 30 40 50 60 70 80 90 100 110 120 130  150];
f0 = 1/(2*pi);

% 创建图形窗口
figure('Position', [100, 100, 1200, 800]);

fprintf('   N     f_max(Hz)    avg_height        rel_error    delta\n');

% 选择几个典型的N值进行可视化
visualize_N = [5 ,10 , 20, 30]; % 要可视化的N值
plot_idx = 1; % 子图索引

for i = 1:length(N_list)
    N = N_list(i);
    
    % 傅里叶合成
    f_fs = zeros(size(t));
    for k = 1:N
        n = 2*k - 1;
        f_fs = f_fs + (4/(n*pi))*sin(n*t);
    end
    
    % 最高频率
    f_max = (2*N-1)*f0;
    
    % 正平台最优区间
    delta = pi/(2*N);%震荡区域
    idx = (t > delta) & (t < pi - delta);
    
    % 平均高度
    avg_height = mean(f_fs(idx));
    rel_error  = avg_height - 1;
    
    
    fprintf('%4d     %8.4f      %.6f     %+.4f      %.4f\n', ...
        N, f_max, avg_height, rel_error,delta);
    
    % 如果当前N在要可视化的列表中，绘制子图
    if ismember(N, visualize_N)
        subplot(2, 2, plot_idx);
        
        % 绘制理想方波（黑色虚线）
        plot(t, f_ideal, 'k-', 'LineWidth', 1.5, 'DisplayName', '理想方波');
        hold on;
        
        % 绘制傅里叶逼近（红色实线）
        plot(t, f_fs, 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('N=%d 逼近', N));
        
        % 标记正平台最优区间
        plot(t(idx), f_fs(idx), 'b.', 'MarkerSize', 6, ...
            'DisplayName', '平台计算区间');
        
        % 添加标题和标签
        title(sprintf('N = %d, f_{max} = %.2f Hz', N, f_max), 'FontSize', 12);
        xlabel('时间 t (rad)', 'FontSize', 11);
        ylabel('幅度', 'FontSize', 11);
        xlim([-pi, pi]);
        ylim([-1.5, 1.5]);
        grid on;
        legend('Location', 'best', 'FontSize', 10);
        
        % 添加平台高度信息
        text(0.5, -1.2, sprintf('平台平均高度: %.6f\n相对误差: %.2e', ...
            avg_height, rel_error), 'FontSize', 10, ...
            'HorizontalAlignment', 'center', 'BackgroundColor', 'white');
        
        hold off;
        plot_idx = plot_idx + 1;
    end
end

% 创建单独的图形展示所有N的逼近效果对比
figure('Position', [100, 100, 1000, 600]);

% 绘制理想方波
plot(t, f_ideal, 'k--', 'LineWidth', 2, 'DisplayName', '理想方波');
hold on;

% 为不同N值选择不同颜色
colors = lines(length(N_list));

% 绘制所有N值的逼近波形
for i = 1:length(N_list)
    N = N_list(i);
    
    % 重新计算该N的傅里叶逼近
    f_fs = zeros(size(t));
    for k = 1:N
        n = 2*k - 1;
        f_fs = f_fs + (4/(n*pi))*sin(n*t);
    end
    
    % 绘制
    plot(t, f_fs, '-', 'LineWidth', 1.2, 'Color', colors(i,:), ...
        'DisplayName', sprintf('N=%d', N));
end

% 添加标题和标签
title('方波傅里叶级数逼近对比 (不同N值)', 'FontSize', 14);
xlabel('时间 t (rad)', 'FontSize', 12);
ylabel('幅度', 'FontSize', 12);
xlim([-pi, pi]);
ylim([-1.5, 1.5]);
grid on;
legend('Location', 'bestoutside', 'FontSize', 10);
hold off;

