%% =========================================
% 方波傅里叶合成动画：显示两个周期
% =========================================
clear; clc; close all;

%% 参数设置
max_N = 50;                     % 最大傅里叶项数
x = linspace(-2*pi, 2*pi, 5000);  % 显示两个周期

%% 理想方波（周期2π，在两个周期内定义）
f_ideal = zeros(size(x));
for i = 1:length(x)
    % 将x映射到[-π, π)区间
    x_mod = mod(x(i) + pi, 2*pi) - pi;
    if x_mod >= 0
        f_ideal(i) = 1;
    else
        f_ideal(i) = -1;
    end
end

%% 预计算所有谐波分量以提高性能
harmonics = zeros(max_N, length(x));
for k = 1:max_N
    n = 2*k - 1;
    harmonics(k, :) = (4/pi) * (1/n) * sin(n * x);
end

%% 创建动画窗口
fig = figure('Position', [100, 100, 900, 600]);

%% 动画循环
S = zeros(1, length(x));  % 初始化部分和
frame_count = 0;
gif_filename = 'square_wave_fourier_2periods.gif';

for N = 1:max_N
    % 添加第N个谐波分量
    S = S + harmonics(N, :);
    
    % 清除并重新绘制
    clf;
    
    % 绘制理想方波和傅里叶逼近
    plot(x, f_ideal, 'k', 'LineWidth', 2); 
    hold on;
    plot(x, S, 'r', 'LineWidth', 1.5);
    grid on;
    
    % 设置x轴以π为单位
    xlim([-2*pi, 2*pi]);
    xticks([-2*pi, -1.5*pi, -pi, -0.5*pi, 0, 0.5*pi, pi, 1.5*pi, 2*pi]);
    xticklabels({'-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π'});
    
    % 标记周期边界
    for k = -2:2
        x_line = k*pi;
        plot([x_line, x_line], [-1.2, 1.2], 'k:', 'LineWidth', 0.5);
        if k ~= 0
            text(x_line, -1.25, sprintf('%dπ', k), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
        end
    end
    
    % 标记间断点
    text(-1.5*pi, -1.35, '间断点位置: -π, 0, π, 2π', ...
        'HorizontalAlignment', 'center', 'FontSize', 10, ...
        'BackgroundColor', 'yellow');
    
    xlabel('x (单位为π)', 'FontSize', 12);
    ylabel('Amplitude', 'FontSize', 12);
    legend('Ideal square wave', ...
           ['Partial sum, N = ', num2str(N)], ...
           'Location', 'best');
    title(sprintf('Gibbs Phenomenon: Square Wave (2周期), N = %d', N), 'FontSize', 14);
    ylim([-1.3, 1.3]);
    
    %% 添加Gibbs过冲分析
    % 在第一个正周期内寻找过冲
    idx = find(x > 0 & x < pi);
    if ~isempty(idx)
        [max_val, max_idx] = max(S(idx));
        x_max = x(idx(max_idx));
        
        plot(x_max, max_val, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        text(x_max, max_val+0.15, sprintf('过冲: %.2f%%', 100*(max_val-1)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    % 显示周期信息
    text(0, -1.2, '周期 T = 2π', ...
        'HorizontalAlignment', 'center', 'FontSize', 11, ...
        'BackgroundColor', 'white');
    
    % 添加当前频率信息
    current_freq = 2*N - 1;
    text(-1.8*pi, 1.25, sprintf('当前谐波频率: %d rad/s', current_freq), ...
        'FontSize', 10, 'BackgroundColor', 'white');
    
    hold off;
    
    % 暂停以控制动画速度
    if N <= 10
        pause(0.3);  % 前10个谐波慢一点
    elseif N <= 20
        pause(0.15);  % 中间谐波中等速度
    else
        pause(0.05);  % 后段谐波快一点
    end
    
    % 捕获帧并保存为GIF
    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    if N == 1
        imwrite(imind, cm, gif_filename, 'gif', ...
            'Loopcount', inf, 'DelayTime', 0.2);
    else
        imwrite(imind, cm, gif_filename, 'gif', ...
            'WriteMode', 'append', 'DelayTime', 0.2);
    end
    
    frame_count = frame_count + 1;
    
    % 每10个谐波保存一张高质量PNG
    if mod(N, 10) == 0
        saveas(fig, sprintf('square_wave_N=%d.png', N));
    end
end

%% 创建对比图：不同N值的合成效果
figure('Position', [100, 100, 1200, 800]);

% 选择几个关键谐波次数
N_values = [1, 3, 5, 10, 20, 50];

for i = 1:length(N_values)
    N = N_values(i);
    
    % 重新计算该N值的傅里叶部分和
    S_compare = zeros(1, length(x));
    for k = 1:N
        n = 2*k - 1;
        S_compare = S_compare + (4/pi) * (1/n) * sin(n * x);
    end
    
    % 绘制子图
    subplot(2, 3, i);
    
    plot(x, f_ideal, 'k', 'LineWidth', 1.5); 
    hold on;
    plot(x, S_compare, 'r', 'LineWidth', 1.5);
    grid on;
    
    % 设置x轴以π为单位
    xlim([-2*pi, 2*pi]);
    xticks([-2*pi, -pi, 0, pi, 2*pi]);
    xticklabels({'-2π', '-π', '0', 'π', '2π'});
    
    % 标记间断点
    for k = -2:2
        x_line = k*pi;
        plot([x_line, x_line], [-1.2, 1.2], 'k:', 'LineWidth', 0.3);
    end
    
    xlabel('x (单位为π)', 'FontSize', 10);
    ylabel('Amplitude', 'FontSize', 10);
    title(sprintf('N = %d', N), 'FontSize', 12);
    ylim([-1.3, 1.3]);
    
    % 计算并显示过冲
    idx = find(x > 0 & x < pi);
    if ~isempty(idx)
        [max_val, max_idx] = max(S_compare(idx));
        overshoot_percent = 100*(max_val-1);
        
        text(0, -1.1, sprintf('过冲: %.2f%%', overshoot_percent), ...
            'HorizontalAlignment', 'center', 'FontSize', 9, ...
            'BackgroundColor', 'white');
    end
    
    hold off;
end

sgtitle('方波傅里叶合成：不同谐波次数对比 (显示两个周期)', 'FontSize', 16);

%% 显示动画信息
fprintf('=========================================\n');
fprintf('方波傅里叶合成动画已生成！\n');
fprintf('=========================================\n');
fprintf('动画文件: %s\n', gif_filename);
fprintf('总帧数: %d\n', frame_count);
fprintf('最大谐波次数: %d\n', max_N);
fprintf('周期: 2π\n');
fprintf('x轴范围: -2π 到 2π (两个周期)\n');
fprintf('生成的PNG文件: square_wave_N=10.png, square_wave_N=20.png, ...\n');
fprintf('\n动画要点:\n');
fprintf('1. 随着谐波次数N增加，合成波形逐渐逼近理想方波\n');
fprintf('2. 在间断点(-π, 0, π, 2π)附近出现Gibbs现象\n');
fprintf('3. 过冲幅度约为9%%，不随N增加而消失\n');
fprintf('4. 振荡频率随N增加而增加\n');
fprintf('=========================================\n');