%% =====================================================
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
phi_c = 0.2*pi;   % 中心
phi_u = 0.6*pi;   % 上
phi_d = 0.5*pi;   % 下
phi_l = 0.3*pi;   % 左
phi_r = 0.4*pi;   % 右

% ---- 矩形尺寸 ----
rect_w = 0.4e-3;   % x 方向宽度
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
theta_x = 1.5*pi/180;
theta_y = 0*pi/180;

R = exp(1i * k * (sin(theta_x) * X + sin(theta_y) * Y));

%% =============================
% 4. 记录全息图 + 1Dfft
% =============================
I = abs(O + R).^2;

H = fftshift(fft(I,[],2),2);
H_amp = abs(H);

%% =============================
% 5. +1 级自动检测（基于 1D FFT）
% =============================
center = N/2 + 1;

% ---- 1D 频谱能量统计（沿 v 方向）----
spec_u = mean(abs(H), 1);   % 对每一列求均值（稳健，抗噪）
spec_u = spec_u(:)';       % 行向量

% ---- 屏蔽零级（DC）----
dc_half_width = 30;         % DC 抑制宽度（像素，可调）
spec_u(center-dc_half_width : center+dc_half_width) = 0;

% ---- 自动寻找 +1 级峰值 ----
[~, u0] = max(spec_u);

% v 方向仍取中心
v0 = center;

fprintf('自动检测 +1 级位置：u0 = %d, v0 = %d\n', u0, v0);

% ---- 可视化检测结果（强烈建议保留）----
figure;
plot(spec_u, 'LineWidth', 1.5); hold on;
plot(u0, spec_u(u0), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
xline(center, '--k');
xlabel('u 方向频率索引');
ylabel('平均幅值');
title('1D FFT 频谱能量 & +1 级自动检测','FontName','SimHei');
grid on;


%% =============================
% 1D 矩形频域滤波器
% =============================
rect_width = 70;    % u 方向宽度（像素）
rect_height = 512;
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
sigma = 20;    % 频域高斯窗口宽度

G = exp(-((U-(u0-center)).^2 + (V-(v0-center)).^2) / (2*sigma^2));

% 频域滤波
Hf = H .* Rect_1D .*G;

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
U = ifft(Hf_center, [], 2);

amp = abs(U);
phi_wrapped = angle(U);

%% =============================
% 8. 最小二乘相位解包裹（替换原相位解包裹）
% =============================
phi_unwrap = least_squares_unwrap_1d_corrected(phi_wrapped, 1); 

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


% 可视化
center_idx = N/2+1 ;
phase_x = phi_unwrap(center_idx , :);  % 第center_idx行的所有列
phase_y = phi_unwrap(:, center_idx);  % 第center_idx列的所有行

x_coord = X(center_idx , :);    % 对应的x坐标（单位：米或毫米）
y_coord = Y(:, center_idx);

figure;
subplot(2,3,1);
surf(X*1000, Y*1000, phi_unwrap, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('原始解包裹相位','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,3,2);% 绘制水平方向相位剖面
plot(x_coord, phase_x, 'b-', 'LineWidth', 2);
xlabel('X 位置 (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('水平方向 (X) 相位剖面', 'FontSize', 14, 'FontName', 'SimHei');
grid on;
xlim([min(x_coord), max(x_coord)]);

subplot(2,3,3);% 绘制垂直方向相位剖面
plot(y_coord, phase_y, 'r-', 'LineWidth', 2);
xlabel('Y 位置 (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('垂直方向 (Y) 相位剖面', 'FontSize', 14, 'FontName', 'SimHei');
grid on;
xlim([min(y_coord), max(y_coord)]);
