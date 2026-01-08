%% =====================================================
%  2D FFT 离轴全息相位重建（+1 级自动检测）+ 最小二乘相位解包裹---高斯窗消除吉布斯
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
phi0 = 2;    %相位=2pi*光程差/波长         
phi_obj = zeros(N);

r0 = 0.6e-3;
phi_obj(X.^2 + Y.^2 <= r0^2) = phi0;


% 2. 构造物体（矩形相位）
rect_width  = 0.8e-3;   % 矩形宽度 (m)
rect_height = 0.4e-3;   % 矩形高度 (m) 与像素设置大小有关
%phi_obj( abs(X) <= rect_width/2 & abs(Y) <= rect_height/2 ) = phi0;
% 3. y
%phi_obj = generate_y_shape_step_phase(N, N, 150, 45, 2);

O = exp(1i*phi_obj);     

%% =============================
% 3. 离轴参考光
% =============================
theta_x = 0*pi/180;
theta_y = 1.8*pi/180;

R = exp(1i * k * (sin(theta_x) * X + sin(theta_y) * Y));

%% =============================
% 4. 记录全息图
% =============================
I = abs(O + R).^2;
%% =============================
% 5. 2D FFT
% =============================
H = fftshift(fft2(I));
H_amp = abs(H);

%% =============================
% 6. 自动检测 +1 级（矩形滤波器）
% =============================
% 去掉中心直流区域（防止选到零级）
center = N/2 + 1;
mask_zero = zeros(N);
mask_zero(center-10:center+10, center-10:center+10) = 1;
H_amp(mask_zero==1) = 0;

% 找频谱最大值 → +1 级
[~, idx] = max(H_amp(:));
[v0, u0] = ind2sub([N,N], idx);
fprintf('自动检测 +1 级位置：u0=%d, v0=%d\n', u0, v0);

% -----------------------------
% 矩形滤波器
% -----------------------------
u = (1:N) - center;
v = (1:N) - center;
[U, V] = meshgrid(u, v);

% ----------- 高斯窗滤波器设计 -----------
sigma = 25;    % 频域高斯窗口宽度

G = exp(-((U-(u0-center)).^2 + (V-(v0-center)).^2) / (2*sigma^2));
Hf = H.*G; 
% 显示原始和滤波信息
figure;
subplot(2,2,1);
imagesc(phi_obj); axis image off;
colormap gray; 
title('真实相位','FontName','SimHei');

subplot(2,2,2);
imagesc(I); axis image off;
colormap gray;
title('离轴全息图','FontName','SimHei');

subplot(2,2,3);
imagesc(log(1+H_amp)); axis image off;
colormap gray;
title('全息图频谱','FontName','SimHei');

subplot(2,2,4);
imagesc(log(1+abs(Hf))); axis image off;
colormap gray;
title('+1 级矩形频谱（自动检测）','FontName','SimHei');

%% =============================
% 7. IFFT 重构复振幅
% =============================
% 计算精准移位量：将(u0,v0)移到频谱矩阵物理中心(center,center)
shift_x = center - u0;  % 列方向移位量（u轴）
shift_y = center - v0;  % 行方向移位量（v轴）

% 替代：用circshift实现频谱中心化（抛弃ifftshift）
% circshift(矩阵, [行移位量, 列移位量]) → 完美匹配u0/v0坐标
Hf_center = circshift(Hf, [shift_y, shift_x]);  

% IFFT重构复振幅（直接对中心化后的频谱操作）
U = ifft2(Hf_center);  

amp = abs(U);
phi_wrapped = angle(U);

figure;
subplot(2,2,1);
imagesc(log(1+abs(Hf_center)));axis image off;
colormap gray;
title('中心化频谱','FontName','SimHei');

subplot(2,2,2);
imagesc(amp); axis image off;
colormap gray;
title('重构幅度','FontName','SimHei');

subplot(2,2,3);
imagesc(phi_wrapped); axis image off;
colormap jet; colorbar;
title('包裹相位','FontName','SimHei');

subplot(2,2,4);
imagesc(phi_wrapped); axis image off;
colormap jet; colorbar;
title('包裹相位','FontName','SimHei');
%% 包裹相位去噪

%% =============================
% 8. 最小二乘相位解包裹（替换原相位解包裹）
% =============================
phi_unwrap = least_squares_unwrap(phi_wrapped);

%% =============================
% 9. 三维显示解包裹相位
% =============================
% 方法1：使用surf函数显示三维曲面
% 方法2：使用mesh函数显示网格图（显示部分数据点，避免过于密集）
figure;
subplot(1,2,1);
imagesc(phi_unwrap); axis image off;
colormap jet; colorbar;
subplot(1,2,2);
title('最小二乘解包裹相位','FontName','SimHei');
skip = 8; % 每隔4个点显示一个，避免图像过于密集
mesh(X(1:skip:end, 1:skip:end)*1000, ...
     Y(1:skip:end, 1:skip:end)*1000, ...
     phi_unwrap(1:skip:end, 1:skip:end));
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('相位 (rad)');
title('解包裹相位三维网格显示 (mesh)','FontName','SimHei');
colormap jet;
colorbar;
view(30, 40);
grid on;
%% =============================
% 9. 倾斜平面校正
% =============================
% 使用背景区域拟合倾斜平面
% 创建背景掩模（假设物体在中心，边缘是背景）
background_mask = zeros(N);
border_width = 30; % 边缘宽度
background_mask(1:border_width, :) = 1;
background_mask(end-border_width+1:end, :) = 1;
background_mask(:, 1:border_width) = 1;
background_mask(:, end-border_width+1:end) = 1;

% 提取背景点的坐标和相位值
bg_points = find(background_mask);
bg_X = X(bg_points);
bg_Y = Y(bg_points);
bg_phi = phi_unwrap(bg_points);

% 最小二乘拟合平面：phi = a*x + b*y + c
A = [bg_X(:), bg_Y(:), ones(length(bg_X), 1)];
coeffs = A \ bg_phi(:); % 求解系数
a = coeffs(1);
b = coeffs(2);
c = coeffs(3);

% 计算拟合的倾斜平面
phi_tilt = a*X + b*Y + c;

% 从解包裹相位中减去倾斜平面
phi_corrected = phi_unwrap - phi_tilt;

% 方法2：使用整个图像拟合平面（作为对比）
A_full = [X(:), Y(:), ones(N*N, 1)];
coeffs_full = A_full \ phi_unwrap(:);
phi_tilt_full = coeffs_full(1)*X + coeffs_full(2)*Y + coeffs_full(3);
phi_corrected_full = phi_unwrap - phi_tilt_full;

%% =============================
% 10. 边缘裁剪
% =============================
crop_pixels = 40; % 裁剪边缘像素数
phi_cropped = phi_corrected(crop_pixels+1:end-crop_pixels, crop_pixels+1:end-crop_pixels);
phi_cropped_full = phi_corrected_full(crop_pixels+1:end-crop_pixels, crop_pixels+1:end-crop_pixels);
X_cropped = X(crop_pixels+1:end-crop_pixels, crop_pixels+1:end-crop_pixels);
Y_cropped = Y(crop_pixels+1:end-crop_pixels, crop_pixels+1:end-crop_pixels);

% 计算裁剪后的真实相位用于比较
phi_obj_cropped = phi_obj(crop_pixels+1:end-crop_pixels, crop_pixels+1:end-crop_pixels);

%% =============================
% 11. 可视化结果
% =============================
% 原始解包裹相位（三维）
figure;
subplot(2,3,1);
surf(X*1000, Y*1000, phi_unwrap, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('原始解包裹相位','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,3,2);
surf(X*1000, Y*1000, phi_tilt, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('拟合倾斜平面','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,3,3);
surf(X*1000, Y*1000, phi_corrected, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('倾斜校正后相位','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,3,4);
imagesc(phi_unwrap); axis image off;
colormap jet; colorbar;
title('原始解包裹相位(2D)','FontName','SimHei');

subplot(2,3,5);
imagesc(phi_corrected); axis image off;
colormap jet; colorbar;
title('倾斜校正后相位(2D)','FontName','SimHei');

subplot(2,3,6);
imagesc(phi_cropped); axis image off;
colormap jet; colorbar;
title('裁剪后校正相位','FontName','SimHei');

% 裁剪后的三维显示
figure;
subplot(2,2,1);
surf(X_cropped*1000, Y_cropped*1000, phi_cropped, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('裁剪后校正相位(3D)','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,2,2);
imagesc(phi_cropped); axis image off;
colormap jet; colorbar;
title('裁剪后校正相位(2D)','FontName','SimHei');

subplot(2,2,3);
surf(X_cropped*1000, Y_cropped*1000, phi_cropped_full, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('全图拟合校正相位(3D)','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,2,4);
imagesc(phi_obj_cropped); axis image off;
colormap jet; colorbar;
title('裁剪后真实相位(参考)','FontName','SimHei');

%% =============================
% 12. 定量评估
% =============================
% 计算相位误差
phase_error = phi_cropped - phi_obj_cropped;
phase_error_full = phi_cropped_full - phi_obj_cropped;

% 计算统计指标
fprintf('===== 相位重建质量评估 =====\n');
fprintf('拟合倾斜平面系数: a=%.6f, b=%.6f, c=%.6f\n', a, b, c);
fprintf('全图拟合倾斜平面系数: a=%.6f, b=%.6f, c=%.6f\n', coeffs_full(1), coeffs_full(2), coeffs_full(3));
fprintf('边缘裁剪像素数: %d\n', crop_pixels);
fprintf('校正后相位范围: %.4f ~ %.4f rad\n', min(phi_cropped(:)), max(phi_cropped(:)));
fprintf('真实相位范围: %.4f ~ %.4f rad\n', min(phi_obj_cropped(:)), max(phi_obj_cropped(:)));
fprintf('背景区域校正RMSE: %.6f rad\n', sqrt(mean(phase_error(:).^2)));
fprintf('全图拟合校正RMSE: %.6f rad\n', sqrt(mean(phase_error_full(:).^2)));

% 显示相位误差分布
figure;
subplot(1,3,1);
imagesc(phase_error); axis image off;
colormap jet; colorbar;
title('背景拟合相位误差','FontName','SimHei');

subplot(1,3,2);
imagesc(phase_error_full); axis image off;
colormap jet; colorbar;
title('全图拟合相位误差','FontName','SimHei');

subplot(1,3,3);
histogram(phase_error(:), 50);
xlabel('相位误差 (rad)'); ylabel('频数');
title('相位误差分布','FontName','SimHei');
grid on;

% 显示边缘区域用于验证背景拟合
%figure;
%imagesc(background_mask); axis image off;
%colormap gray;
%title('背景区域掩模（白色为背景）','FontName','SimHei');

%% =============================
% X方向（水平）一维相位剖面
% =============================
% 假设已有校正裁剪后的相位图 phi_cropped 和坐标 X_cropped, Y_cropped
% 尺寸：Nc × Nc（裁剪后的尺寸）

Nc = size(phi_cropped, 1);
center_idx = floor(Nc/2) + 1;  % 中心索引

% 提取水平中心线
phase_x = phi_cropped(center_idx, :);  % 第center_idx行的所有列
x_coord = X_cropped(center_idx, :);    % 对应的x坐标（单位：米或毫米）

% 转换为毫米显示
x_coord_mm = x_coord * 1000;  % 米→毫米

% 绘制水平方向相位剖面
figure('Position', [100, 100, 800, 400]);
subplot(1,2,1);
plot(x_coord_mm, phase_x, 'b-', 'LineWidth', 2);
xlabel('X 位置 (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('水平方向 (X) 相位剖面', 'FontSize', 14, 'FontName', 'SimHei');
grid on;
xlim([min(x_coord_mm), max(x_coord_mm)]);

% 在原图中标记提取线
subplot(1,2,2);
imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phi_cropped);
hold on;
plot(x_coord_mm, zeros(size(x_coord_mm)), 'r-', 'LineWidth', 3);
xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
title('提取线位置', 'FontSize', 14, 'FontName', 'SimHei');
colormap jet; colorbar;
axis image;
% ============================
% Y方向（垂直）一维相位剖面
% =============================
% 提取垂直中心线
phase_y = phi_cropped(:, center_idx);  % 第center_idx列的所有行
y_coord = Y_cropped(:, center_idx);    % 对应的y坐标

% 转换为毫米显示
y_coord_mm = y_coord * 1000;

% 绘制垂直方向相位剖面
figure('Position', [100, 100, 800, 400]);
subplot(1,2,1);
plot(y_coord_mm, phase_y, 'r-', 'LineWidth', 2);
xlabel('Y 位置 (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('垂直方向 (Y) 相位剖面', 'FontSize', 14, 'FontName', 'SimHei');
grid on;
xlim([min(y_coord_mm), max(y_coord_mm)]);

% 在原图中标记提取线
subplot(1,2,2);
imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phi_cropped);
hold on;
plot(zeros(size(y_coord_mm)), y_coord_mm, 'g-', 'LineWidth', 3);
xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
title('提取线位置', 'FontSize', 14, 'FontName', 'SimHei');
colormap jet; colorbar;
axis image;
%% =============================
% 45度方向一维相位剖面（两种实现方法）
% =============================

% 方法A：直接提取主对角线（简单但有限制）
if Nc == size(phi_cropped, 2)  % 确保是正方形
    phase_diag1 = diag(phi_cropped);  % 主对角线
    phase_diag2 = diag(fliplr(phi_cropped));  % 副对角线
    
    % 计算对角线上的物理距离
    diag_coord = sqrt(2) * X_cropped(1,:);  % 对角线距离
    diag_coord_mm = diag_coord * 1000;
    
    % 绘制两条对角线相位
    figure('Position', [100, 100, 800, 400]);
    subplot(1,2,1);
    plot(diag_coord_mm, phase_diag1, 'm-', 'LineWidth', 2, 'DisplayName', '主对角线');
    hold on;
    plot(diag_coord_mm, phase_diag2, 'c-', 'LineWidth', 2, 'DisplayName', '副对角线');
    xlabel('对角线位置 (mm)', 'FontSize', 12, 'FontName', 'SimHei');
    ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
    title('45度方向相位剖面', 'FontSize', 14, 'FontName', 'SimHei');
    legend('show', 'Location', 'best');
    grid on;
    
    % 在原图中标记提取线
    subplot(1,2,2);
    imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phi_cropped);
    hold on;
    plot([min(X_cropped(:)), max(X_cropped(:))]*1000, ...
         [min(Y_cropped(:)), max(Y_cropped(:))]*1000, 'y-', 'LineWidth', 3);
    plot([min(X_cropped(:)), max(X_cropped(:))]*1000, ...
         [max(Y_cropped(:)), min(Y_cropped(:))]*1000, 'y-', 'LineWidth', 3);
    xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
    ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
    title('提取线位置', 'FontSize', 14, 'FontName', 'SimHei');
    colormap jet; colorbar;
    axis image;
end

% 在原图中标记提取线
subplot(1,2,2);
imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phi_cropped);
hold on;

% 绘制45度线
center_x = mean(X_cropped(:))*1000;
center_y = mean(Y_cropped(:))*1000;
len = max(abs([X_cropped(:); Y_cropped(:)]))*1000;
x_line_45 = [center_x - len, center_x + len];
y_line_45 = [center_y - len, center_y + len];
plot(x_line_45, y_line_45, 'w--', 'LineWidth', 2);
xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
title('45度提取线位置', 'FontSize', 14, 'FontName', 'SimHei');
colormap jet; colorbar;
axis image;


%% =============================
% 综合对比：原始相位与重建相位剖面分析
% ============================

% 检查变量是否存在
if ~exist('phi_obj_cropped', 'var') || ~exist('phi_cropped', 'var')
    error('请先运行之前的代码，确保phi_obj_cropped和phi_cropped存在');
end

%% =============================
% 1. 提取四个方向的相位剖面
% =============================

% 获取裁剪后图像的尺寸
[Ny, Nx] = size(phi_cropped);
center_idx = floor(Ny/2) + 1;  % 中心索引

% 1.1 原始相位 - X方向剖面
phase_orig_x = phi_obj_cropped(center_idx, :);
x_coord = X_cropped(center_idx, :) * 1000;  % 转换为毫米

% 1.2 重建相位 - X方向剖面
phase_rec_x = phi_cropped(center_idx, :);

% 1.3 重建相位 - Y方向剖面
phase_rec_y = phi_cropped(:, center_idx);
y_coord = Y_cropped(:, center_idx) * 1000;

% 1.4 重建相位 - 45度方向剖面
% 使用之前定义的通用函数提取45度剖面
if ~exist('extract_phase_profile', 'file')
    % 如果函数不存在，定义它
    phase_rec_45 = diag(phi_cropped);  % 临时使用主对角线
    dist_45 = sqrt(2) * X_cropped(center_idx, :) * 1000;
else
    [phase_rec_45, dist_45] = extract_phase_profile(phi_cropped, X_cropped, Y_cropped, 45);
end

%% =============================
% 2. 坐标归一化处理（便于比较）
% =============================

% 将所有坐标归一化到[-1, 1]区间
x_norm = (x_coord - mean(x_coord)) / max(abs(x_coord - mean(x_coord)));
y_norm = (y_coord - mean(y_coord)) / max(abs(y_coord - mean(y_coord)));

% 归一化45度方向坐标
if exist('dist_45', 'var')
    dist_45_norm = (dist_45 - mean(dist_45)) / max(abs(dist_45 - mean(dist_45)));
else
    dist_45_norm = x_norm;  % 如果不存在，使用x_norm
end

%% =============================
% 3. 绘制综合对比图（一个图四条线）
% =============================

figure('Position', [100, 100, 1200, 500]);

% 子图1：四条线综合对比
subplot(1, 2, 1);
hold on;

% 1. 原始相位 - X方向（黑色实线）
plot(x_norm, phase_orig_x, 'k-', 'LineWidth', 3, 'DisplayName', '原始相位 (X方向)');

% 2. 重建相位 - X方向（蓝色虚线）
plot(x_norm, phase_rec_x, 'b--', 'LineWidth', 2.5, 'DisplayName', '重建相位 (X方向)');

% 3. 重建相位 - Y方向（红色点划线）
plot(y_norm, phase_rec_y, 'r-.', 'LineWidth', 2.5, 'DisplayName', '重建相位 (Y方向)');

% 4. 重建相位 - 45度方向（绿色点线）
if length(phase_rec_45) == length(dist_45_norm)
    plot(dist_45_norm, phase_rec_45, 'g:', 'LineWidth', 2.5, 'DisplayName', '重建相位 (45度方向)');
else
    % 如果长度不匹配，可能需要重新采样
    fprintf('注意: 45度方向长度不匹配，进行插值处理\n');
    phase_rec_45_interp = interp1(linspace(-1, 1, length(phase_rec_45)), ...
                                  phase_rec_45, x_norm, 'linear');
    plot(x_norm, phase_rec_45_interp, 'g:', 'LineWidth', 2.5, 'DisplayName', '重建相位 (45度方向)');
end

% 图形美化
xlabel('归一化位置', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('原始相位与重建相位剖面综合对比', 'FontSize', 14, 'FontName', 'SimHei');
legend('show', 'Location', 'best', 'FontSize', 10, 'FontName', 'SimHei');
grid on;
xlim([-1, 1]);
hold off;

% 子图2：相位误差分析
subplot(1, 2, 2);
hold on;

% 计算各方向的相位误差（重建-原始）
% 注意：需要确保坐标对齐，这里假设都能对齐到x_norm坐标

% X方向误差
if length(phase_rec_x) == length(phase_orig_x)
    error_x = phase_rec_x - phase_orig_x;
    plot(x_norm, error_x, 'b-', 'LineWidth', 2, 'DisplayName', 'X方向误差');
end

% Y方向误差（需要插值到x_norm坐标）
phase_rec_y_interp = interp1(y_norm, phase_rec_y, x_coord, 'linear');
phase_orig_y = phi_obj_cropped(:, center_idx)';
phase_orig_y_interp = interp1(y_coord, phase_orig_y, x_coord, 'linear');
error_y = phase_rec_y_interp - phase_orig_y_interp;
plot(x_norm, error_y, 'r-', 'LineWidth', 2, 'DisplayName', 'Y方向误差');

% 45度方向误差（需要插值到x_norm坐标）
if exist('phase_rec_45', 'var')
    if length(phase_rec_45) == length(dist_45_norm)
        phase_rec_45_interp = interp1(dist_45_norm, phase_rec_45, x_coord, 'linear');
        % 原始相位在45度方向也需要提取
        if exist('extract_phase_profile', 'file')
            [phase_orig_45, ~] = extract_phase_profile(phi_obj_cropped, X_cropped, Y_cropped, 45);
            phase_orig_45_interp = interp1(dist_45_norm, phase_orig_45, x_coord, 'linear');
        else
            phase_orig_45 = diag(phi_obj_cropped);
            phase_orig_45_interp = interp1(x_coord, phase_orig_45, x_coord, 'linear');
        end
        error_45 = phase_rec_45_interp - phase_orig_45_interp;
        plot(x_norm, error_45, 'g-', 'LineWidth', 2, 'DisplayName', '45度方向误差');
    end
end

% 误差分析图美化
xlabel('归一化位置', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位误差 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('重建相位误差分析', 'FontSize', 14, 'FontName', 'SimHei');
legend('show', 'Location', 'best', 'FontSize', 10, 'FontName', 'SimHei');
grid on;
xlim([-1, 1]);
hold off;

%% =============================
% 4. 绘制原始相位与重建相位的二维对比
% =============================

figure('Position', [100, 100, 1000, 400]);

% 子图1：原始相位
subplot(1, 3, 1);
imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phi_obj_cropped);
hold on;
% 标记剖面线位置
plot(x_coord, zeros(size(x_coord)), 'w-', 'LineWidth', 2);  % X方向
plot(zeros(size(y_coord)), y_coord, 'w-', 'LineWidth', 2);  % Y方向
len = max(abs([X_cropped(:); Y_cropped(:)]))*1000;
plot([-len, len], [-len, len], 'w-', 'LineWidth', 2);  % 45度方向
xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
title('原始相位 (参考)', 'FontSize', 14, 'FontName', 'SimHei');
colormap jet; colorbar;
axis image;

% 子图2：重建相位
subplot(1, 3, 2);
imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phi_cropped);
hold on;
% 标记剖面线位置
plot(x_coord, zeros(size(x_coord)), 'w-', 'LineWidth', 2);  % X方向
plot(zeros(size(y_coord)), y_coord, 'w-', 'LineWidth', 2);  % Y方向
plot([-len, len], [-len, len], 'w-', 'LineWidth', 2);  % 45度方向
xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
title('重建相位', 'FontSize', 14, 'FontName', 'SimHei');
colormap jet; colorbar;
axis image;

% 子图3：相位误差图
subplot(1, 3, 3);
phase_error_2d = phi_cropped - phi_obj_cropped;
imagesc(X_cropped(1,:)*1000, Y_cropped(:,1)'*1000, phase_error_2d);
xlabel('X (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('Y (mm)', 'FontSize', 12, 'FontName', 'SimHei');
title('相位误差 (重建-原始)', 'FontSize', 14, 'FontName', 'SimHei');
colormap jet; colorbar;
caxis([-0.5, 0.5]);  % 限制颜色范围，便于观察
axis image;

%% =============================
% 5. 定量分析与统计
% =============================

fprintf('\n===== 相位重建质量定量分析 =====\n\n');

% 计算全局统计
phase_error_2d_flat = phase_error_2d(:);
rmse = sqrt(mean(phase_error_2d_flat.^2));
max_error = max(abs(phase_error_2d_flat));
mean_error = mean(phase_error_2d_flat);
std_error = std(phase_error_2d_flat);

fprintf('全局统计:\n');
fprintf('  均方根误差 (RMSE): %.6f rad\n', rmse);
fprintf('  最大绝对误差: %.6f rad\n', max_error);
fprintf('  平均误差: %.6f rad\n', mean_error);
fprintf('  误差标准差: %.6f rad\n\n', std_error);

% 计算各方向误差统计
fprintf('各方向误差统计:\n');
if exist('error_x', 'var')
    fprintf('  X方向:\n');
    fprintf('    RMSE: %.6f rad\n', sqrt(mean(error_x.^2)));
    fprintf('    最大误差: %.6f rad\n', max(abs(error_x)));
end

if exist('error_y', 'var')
    fprintf('  Y方向:\n');
    fprintf('    RMSE: %.6f rad\n', sqrt(mean(error_y.^2)));
    fprintf('    最大误差: %.6f rad\n', max(abs(error_y)));
end

if exist('error_45', 'var')
    fprintf('  45度方向:\n');
    fprintf('    RMSE: %.6f rad\n', sqrt(mean(error_45.^2)));
    fprintf('    最大误差: %.6f rad\n', max(abs(error_45)));
end

% 计算相位保真度（相关系数）
corr_coeff = corr2(phi_cropped, phi_obj_cropped);
fprintf('\n相位保真度:\n');
fprintf('  相关系数: %.6f\n', corr_coeff);

% 计算结构相似性指数 (SSIM)
if exist('ssim', 'file') || exist('ssim_index', 'file')
    try
        % 尝试使用Image Processing Toolbox的ssim函数
        ssim_val = ssim(phi_cropped, phi_obj_cropped);
        fprintf('  结构相似性指数 (SSIM): %.6f\n', ssim_val);
    catch
        % 如果ssim不可用，使用自定义计算
        fprintf('  SSIM: 需要Image Processing Toolbox\n');
    end
else
    fprintf('  SSIM: 需要Image Processing Toolbox\n');
end


