%%=====================================================
%  1D FFT 离轴全息相位重建（+1 级自动检测）+ 最小二乘相位解包裹---高斯窗消除吉布斯
% =====================================================
clear; clc; close all;
%% =============================
% 1. 参数设置
% =============================
N = 512;                 % 图像尺寸
dx = 3.5e-6;             % 像元尺寸 (m)
lambda = 632.8e-9;       % 波长
k = 2*pi/lambda;

x = (-N/2:N/2-1)*dx;
[X,Y] = meshgrid(x,x);
%% =============================
% 2. 构造物体（圆形相位）
% =============================
phi0 = 1.5;    %相位=2pi*光程差/波长         

% r0 = 0.6e-3;
% phi_obj(X.^2 + Y.^2 <= r0^2) = phi0;
% phi_obj = generate_y_shape_step_phase(N, N, 150, 45, 2);
phi_obj = zeros(N);

% ---- 相位值 ----
phi_c = 0.4*pi;   % 中
phi_u = 0.6*pi;   % 上
phi_d = 0.5*pi;   % 下
phi_l = 0.55*pi;   % 左 
phi_r = 0.45*pi;   % 右 最右边的原始相位是1.2566 中心是1.2543 插值是0.0023

% ---- 矩形尺寸 ----
rect_w = 0.3e-3;   % x 方向宽度
rect_h = 0.3e-3;   % y 方向高度
% ---- 中心间距 ----
d = 0.55e-3;

% ---- 中心矩形 ----
phi_obj( abs(X)<=rect_w/2 & abs(Y)<=rect_h/2 ) = phi_c;

% ---- 上矩形 ----
phi_obj( abs(X)<=rect_w/2 & abs(Y-d)<=rect_h/2 ) = phi_u;

% ---- 下矩形 ----
phi_obj( abs(X)<=rect_w/2 & abs(Y+d)<=rect_h/2 ) = phi_d;

% ---- 左矩形 ----
phi_obj( abs(X+d)<=rect_w/2 & abs(Y)<=rect_h/2 ) = phi_l;

% ---- 右矩形 ----
phi_obj( abs(X-d)<=rect_w/2 & abs(Y)<=rect_h/2 ) = phi_r;

O = exp(1i * phi_obj);

%% =============================
% 3. 离轴参考光
% =============================
theta_x = 0*pi/180;
theta_y = 1.5*pi/180;

R = exp(1i * k * (sin(theta_x) * X + sin(theta_y) * Y));

%% =============================
% 4. 记录全息图 + 1Dfft
% =============================
I = abs(O + R).^2;

H = fftshift(fft(I,[],1),1);
H_amp = abs(H);

%% =============================
% 5. +1 级自动检测（基于 1D FFT，垂直方向）
% =============================
center = N/2 + 1;

% ---- 对每一行做 FFT 能量统计 ----
spec_v = mean(abs(H), 2);    % 对每一行求均值（垂直方向投影）
spec_v = spec_v(:);           % 列向量

% ---- 屏蔽零级（DC）----
dc_half_width = 30;           % DC 抑制宽度（像素，可调）
spec_v(center-dc_half_width : center+dc_half_width) = 0;

% ---- 自动寻找 +1 级峰值 ----
[~, v0] = max(spec_v);

% u 方向仍取中心
u0 = center;
fprintf('垂直方向自动检测 +1 级位置：u0 = %d, v0 = %d\n', u0, v0);

% ---- 可视化垂直方向频谱 ----
figure;
plot(spec_v, 'LineWidth', 1.5); hold on;
plot(v0, spec_v(v0), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
xline(center, '--k');
xlabel('v 方向频率索引');
ylabel('平均幅值');
title('垂直方向 1D FFT 频谱能量 & +1 级自动检测','FontName','SimHei');
grid on;

%% =============================
% 1D 矩形频域滤波器
% =============================
rect_width = 512;    % u 方向宽度（像素）
rect_height = 100;
u = (1:N) - center;
v = (1:N) - center;
[U, V] = meshgrid(u, v); 
% +1 级中心在频域中的坐标（以中心为零频） 
u_c = u0 - center; 
v_c = v0 - center; 
% ============================= 
% 矩形频域滤波器
% ============================= 
Rect_1D = zeros(N); 
Rect_1D( abs(U - u_c) <= rect_width/2 & ... 
    abs(V - v_c) <= rect_height/2 ) = 1;
% 扩展为 2D（对所有 y 行相同）
%Rect_1D = repmat(Rect_1D, N, 1);
% ----------- 高斯窗滤波器设计 -----------
sigma = 28;    % 频域高斯窗口宽度
G = exp(-((U-(u0-center)).^2 + (V-(v0-center)).^2) / (2*sigma^2));

% 频域滤波
Hf = H .* Rect_1D .* G;%32是sigima = 0

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
% 7. IFFT 重构复振幅（自动频谱中心化）
% =============================

% ---- 根据 +1 级位置自动计算平移量 ----
shift_x = center - u0;   % u 方向（列）
shift_y = center - v0;   % v 方向（行）

fprintf('频谱自动中心化位移：shift_x = %d, shift_y = %d\n', shift_x, shift_y);

% ---- 频谱平移到中心 ----
Hf_center = circshift(Hf, [shift_y, shift_x]);

% ---- 1D IFFT 重构 ----
U = ifft(Hf_center, [], 1);
amp = abs(U);
phi_wrapped = angle(U);

%% =============================
% 8. 1D最小二乘相位解包裹（替换原相位解包裹）
% =============================
phi_unwrap = least_squares_unwrap_1d_corrected(phi_wrapped, 2);

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
title('包裹相位（1D 重建）','FontName','SimHei');

subplot(2,2,4);
imagesc(phi_unwrap); axis image off;
colormap jet; colorbar;
title('1D 解包裹相位（沿 Y）','FontName','SimHei');
pixel_idx = 1:N;   % 像素索引（1 ~ N）

% 计算相位误差
phase_error = phi_unwrap - phi_obj;

% 计算统计指标
fprintf('背景区域校正RMSE: %.6f rad\n', sqrt(mean(phase_error(:).^2)));
% 可视化
%edge_offset = round(rect_h/(2*dx));  % 半高对应的像素数
center_idx = N/2+1 ;
edge_offset = round(rect_h/(2*dx));
phase_x = phi_unwrap(center_idx , :); 
phase_x_edge = phi_unwrap(center_idx+ edge_offset-8, :);  % 第center_idx行的所有列

phase_y = phi_unwrap(:, center_idx);  % 第center_idx列的所有行

x_coord = X(center_idx , :);    % 对应的x坐标（单位：米或毫米）
y_coord = Y(:, center_idx);

figure;
plot(pixel_idx, phase_x, 'b-', 'LineWidth', 2); hold on;
plot(pixel_idx, phase_x_edge, 'r--', 'LineWidth', 2);

xlabel('X 方向像素索引', 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontName', 'SimHei');
title('X方向相位剖面对比：台阶中间 vs 台阶边缘', 'FontName', 'SimHei');

legend('台阶中间剖面', '台阶边缘剖面', 'FontName', 'SimHei');
grid on;
xlim([1 N]);
%% =============================
% 9. 分区滤波（1D，仅沿 Y 方向，正确版）
% =============================
T = 0.2*pi;
% -------- 二值分区 --------
mask_edge_bin = phi_unwrap > T;
% -------- mask 的 1D Y 向模糊 --------
sigma_mask = 3;
len = ceil(6*sigma_mask);
if mod(len,2)==0, len = len+1; end
h_mask = fspecial('gaussian', [len 1], sigma_mask);

mask_edge = imfilter(double(mask_edge_bin), h_mask, 'replicate');
mask_edge = min(max(mask_edge, 0), 1);
mask_smooth = 1 - mask_edge;

% -------- 相位的 1D Y 向分区滤波 --------
sigma_smooth = 3.5; %底部
sigma_edge   = 2.0; %定

h_s = fspecial('gaussian', [ceil(6*sigma_smooth)+1 1], sigma_smooth);
h_e = fspecial('gaussian', [ceil(6*sigma_edge)+1   1], sigma_edge);

phi_smooth = imfilter(phi_unwrap, h_s, 'replicate');
phi_edge   = imfilter(phi_unwrap, h_e, 'replicate');

% -------- 软融合 --------
phi_filt = mask_smooth .* phi_smooth + mask_edge .* phi_edge;

%% =============================
% 可视化结果
% =============================
figure;
subplot(1,3,1);
imagesc(phi_unwrap); axis image off;
colormap jet; colorbar;
title('原始解包裹相位','FontName','SimHei');

subplot(1,3,2);
imagesc(phi_filt); axis image off;
colormap jet; colorbar;
title('两区域滤波后相位','FontName','SimHei');

subplot(1,3,3);
imagesc(phi_filt - phi_unwrap); axis image off;
colormap jet; colorbar;
title('滤波改变量','FontName','SimHei');

%% =============================
% 10. 中心水平线对比
% =============================
center_row = round(N/2);   % 中心水平线索引
edge_offset = round(rect_h/(2*dx));
% 提取中心行的相位
phi_center_orig = phi_unwrap(:, center_row);
phi_center_filt = phi_filt(:, center_row);

% 横坐标（x 轴）
x_axis = x;  % 已经在前面定义 x = (-N/2:N/2-1)*dx

% 绘图对比
figure;
plot(x_axis, phi_center_orig, '-b', 'LineWidth', 1.5); hold on;
plot(x_axis, phi_center_filt, '-r', 'LineWidth', 1.5);
xlabel('mm');
ylabel('解包裹相位 [rad]');
legend('原始相位','两区域滤波后', 'FontName', 'SimHei');
grid on;
title('过中心垂直线（Y）的相位对比','FontName','SimHei');
%% =============================
% 12. 中心水平线（X方向）对比（已有中心行代码，可调整美观）
% =============================
center_row = round(N/2);   % 中心行索引

% 提取中心行的相位
phi_center_horiz_orig = phi_unwrap(center_row, :);
phi_center_horiz_filt = phi_filt(center_row, :);

% X坐标（单位 mm）
x_axis = x * 1e3;

% 绘图
figure;
plot(x_axis, phi_center_horiz_orig, '-b', 'LineWidth', 1.5); hold on;
plot(x_axis, phi_center_horiz_filt, '-r', 'LineWidth', 1.5);
xlabel('x [mm]');
ylabel('解包裹相位 [rad]');
legend('原始相位','两区域滤波后','FontName','SimHei');
grid on;
title('过中心水平线的相位对比（X方向）','FontName','SimHei');


%% =============================
% 13. 三维相位对比（原始 vs 分区滤波）
% =============================

% 为了显示更平滑，适当降采样（可选）
ds = 2;   % 降采样因子，1 表示不降采样
X_ds = X(1:ds:end, 1:ds:end) * 1e3;   % mm
Y_ds = Y(1:ds:end, 1:ds:end) * 1e3;
phi_unwrap_ds = phi_unwrap(1:ds:end, 1:ds:end);
phi_filt_ds   = phi_filt(1:ds:end, 1:ds:end);

%% ---- 原始解包裹相位（三维）----
figure;
surf(X_ds, Y_ds, phi_unwrap_ds, ...
     'EdgeColor','none');
colormap jet;
colorbar;
xlabel('x [mm]');
ylabel('y [mm]');
zlabel('Phase [rad]');
title('原始解包裹相位（三维）','FontName','SimHei');
view(45,30);
lighting phong;
camlight headlight;
axis tight;

%% ---- 分区滤波后相位（三维）----
figure;
surf(X_ds, Y_ds, phi_filt_ds, ...
     'EdgeColor','none');
colormap jet;
colorbar;
xlabel('x [mm]');
ylabel('y [mm]');
zlabel('Phase [rad]');
title('分区滤波后相位（三维）','FontName','SimHei');
view(45,30);
lighting phong;
camlight headlight;
axis tight;


