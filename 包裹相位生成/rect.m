%% =====================================================
%  2D FFT 离轴全息相位重建（+1 级自动检测）+ 最小二乘相位解包裹
% =====================================================
clear; clc; close all;
%% =============================
% 1. 参数设置
% =============================
N = 512;                 % 图像尺寸
dx = 6.5e-6;             % 像元尺寸 (m)
lambda = 632.8e-9;       % 波长
k = 2*pi/lambda;

x = (-N/2:N/2-1)*dx;
[X,Y] = meshgrid(x,x);


%% =============================
% 2. 构造物体（圆形相位）
% =============================
r0 = 0.6e-3; 
phi0 = 1;  %相位=2pi*光程差/波长         

phi_obj = zeros(N);
phi_obj(X.^2 + Y.^2 <= r0^2) = phi0;

O = exp(1i*phi_obj);     

%% =============================
% 3. 离轴参考光
% =============================
theta_x = 0*pi/180;
theta_y = 0.8*pi/180;

R = exp(1i * k * (sin(theta_x) * X + sin(theta_y) * Y));

%% =============================
% 1.1 窗函数定义（用于对比）
% =============================
W_set = {};
W_name = {};

% 不加窗（矩形窗）
W_set{end+1}  = ones(N);
W_name{end+1} = 'No window (Rectangular)';

% Hann 窗
W_set{end+1}  = hann(N) * hann(N).';
W_name{end+1} = 'Hann-汉宁';

% Hamming 窗
W_set{end+1}  = hamming(N) * hamming(N).';
W_name{end+1} = 'Hamming-汉明';

% Gaussian 窗
sigma = 0.1 * N;
[xg, yg] = meshgrid(1:N, 1:N);
c = (N+1)/2;
W_set{end+1}  = exp(-((xg-c).^2 + (yg-c).^2)/(2*sigma^2));
W_name{end+1} = 'Gaussian-高斯';

num_case = length(W_set);
%% =============================
% 窗函数对比主循环
% =============================
% =============================
% 三方向相位剖面对比图
% =============================
figure;
set(gcf,'Position',[100 100 1200 400]);

color_set = lines(num_case);

subplot(1,3,1); hold on; grid on; title('X direction');
subplot(1,3,2); hold on; grid on; title('Y direction');
subplot(1,3,3); hold on; grid on; title('45° direction');
% 存储RMSE
STD_all = zeros(1, num_case);

for icase = 1:num_case

    fprintf('\n=============================\n');
    fprintf('当前窗函数：%s\n', W_name{icase});
    fprintf('=============================\n');

    %% =============================
    % 4. 记录全息图 + 加窗
    % =============================
    I = abs(O + R).^2;
    I = I .* W_set{icase};

    %% =============================
    % 5. 2D FFT
    % =============================
    H = fftshift(fft2(I));
    H_amp = abs(H);

    %% =============================
    % 6. 自动检测 +1 级（与你原代码完全一致）
    % =============================
    center = N/2 + 1;
    H_amp(center-10:center+10, center-10:center+10) = 0;
    [~, idx] = max(H_amp(:));
    [v0, u0] = ind2sub([N,N], idx);

    % -----------------------------
    % 矩形滤波器
    % -----------------------------
    u = (1:N) - center;
    v = (1:N) - center;
    [U, V] = meshgrid(u, v);

    % ----------- 高斯窗滤波器设计 -----------
    sigma =15;    % 高斯窗标准差，控制平滑程度
    G = exp(-((U-(u0-center)).^2 + (V-(v0-center)).^2) / (2*sigma^2));
    Hf = H.*G; 

    %% =============================
    % 7. 频谱中心化 + IFFT
    % =============================
    shift_x = center - u0;
    shift_y = center - v0;
    Hf_center = circshift(Hf,[shift_y,shift_x]);
    U = ifft2(Hf_center);

    phi_wrapped = angle(U);

    %% =============================
    % 8. 最小二乘解包裹
    % =============================
    phi_unwrap = least_squares_unwrap(phi_wrapped);

    %% =============================
    % 9. 倾斜平面校正（背景拟合）
    % =============================
    background_mask = zeros(N);
    bw = 30;
    background_mask(1:bw,:) = 1;
    background_mask(end-bw+1:end,:) = 1;
    background_mask(:,1:bw) = 1;
    background_mask(:,end-bw+1:end) = 1;

    idx_bg = find(background_mask);
    A = [X(idx_bg), Y(idx_bg), ones(length(idx_bg),1)];
    coeff = A \ phi_unwrap(idx_bg);

    phi_tilt = coeff(1)*X + coeff(2)*Y + coeff(3);
    phi_corr = phi_unwrap - phi_tilt;

    %% =============================
    % 10. 裁剪
    % =============================
    crop = 40;
    phi_cropped = phi_corr(crop+1:end-crop, crop+1:end-crop);
    phi_true_cropped = phi_obj(crop+1:end-crop, crop+1:end-crop);
    phi = phi_cropped;
    [Nx, Ny] = size(phi);
    cx = round(Nx/2);
    cy = round(Ny/2);

    %% ========== X 方向 ==========
    x_idx = 1:Ny;
    phi_x = phi(cy,:);
    px = polyfit(x_idx, phi_x, 1);
    phi_x_fit = polyval(px, x_idx);
    phi_true_x = phi_true_cropped(cy,:);

    subplot(1,3,1);
    plot(x_idx, phi_x, ...
        'LineWidth',2, ...
        'Color',color_set(icase,:), ...
        'DisplayName',W_name{icase});
    plot(x_idx, phi_x_fit,'--', ...
        'Color',color_set(icase,:), ...
        'HandleVisibility','off');

    %% ========== Y 方向 ==========
    y_idx = 1:Nx;
    phi_y = phi(:,cx);
    py = polyfit(y_idx, phi_y.', 1);
    phi_y_fit = polyval(py, y_idx);

    subplot(1,3,2);
    plot(y_idx, phi_y, ...
        'LineWidth',2, ...
        'Color',color_set(icase,:), ...
        'DisplayName',W_name{icase});
    plot(y_idx, phi_y_fit,'--', ...
        'Color',color_set(icase,:), ...
        'HandleVisibility','off');

    %% ========== 45° 方向 ==========
    n_diag = min(Nx,Ny);
    idx = 1:n_diag;
    phi_d = diag(phi);
    pd = polyfit(idx, phi_d.', 1);
    phi_d_fit = polyval(pd, idx);

    subplot(1,3,3);
    plot(idx, phi_d, ...
        'LineWidth',2, ...
        'Color',color_set(icase,:), ...
        'DisplayName',W_name{icase});
    plot(idx, phi_d_fit,'--', ...
        'Color',color_set(icase,:), ...
        'HandleVisibility','off');


    %% =============================
    % 11. 标准差 计算
    % =============================
    err = phi_cropped - phi_true_cropped;
    err_mean = mean(err(:));
    STD_all(icase) = std(err(:));  % 使用MATLAB内置的std函数

    fprintf('标准差 = %.6f rad\n', STD_all(icase));

end

subplot(1,3,1);
xlabel('Pixel'); ylabel('Phase (rad)');
legend('Location','best');

subplot(1,3,2);
xlabel('Pixel'); ylabel('Phase (rad)');

subplot(1,3,3);
xlabel('Pixel'); ylabel('Phase (rad)');

sgtitle('Phase profiles and tangent lines under different window functions');

%% =============================
% 添加窗函数对比总结图
% =============================
figure;
set(gcf,'Position',[100 100 800 600]);

% 创建子图布局
subplot(2,3,1); imagesc(W_set{1}); title(W_name{1}); axis image; colorbar;
subplot(2,3,2); imagesc(W_set{2}); title(W_name{2}); axis image; colorbar;
subplot(2,3,3); imagesc(W_set{3}); title(W_name{3}); axis image; colorbar;
subplot(2,3,4); imagesc(W_set{4}); title(W_name{4}); axis image; colorbar;

colormap(jet);
sgtitle('2D Window Functions');

% 显示标准差对比图
figure;
bar(STD_all);
set(gca, 'XTickLabel', W_name, 'XTickLabelRotation', 45);
ylabel('标准差 (rad)');
title('不同窗函数下相位重建的标准差对比');
grid on;

% 在柱状图上显示数值
for i = 1:length(STD_all)
    text(i, STD_all(i)+0.001, sprintf('%.4f', STD_all(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
ylim([0, max(STD_all)*1.2]);