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
noise = 0.02;
noise_pha = noise * randn(N,N);
%% =============================
% 2. 构造物体（圆形相位）
% =============================
phi0 = 1.5;    %相位=2pi*光程差/波长         

% r0 = 0.6e-3;
% phi_obj(X.^2 + Y.^2 <= r0^2) = phi0;
% phi_obj = generate_y_shape_step_phase(N, N, 150, 45, 2);
phi_obj = zeros(N);

% ---- 相位值 ----
phi_c = 0.2*pi;   % 中 0.6283
phi_u = 0.6*pi;   % 上 1.8850
phi_d = 0.5*pi;   % 下 1.5708
phi_l = 0.3*pi;   % 左 0.9425
phi_r = 0.4*pi;   % 右 1.2566

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

O = exp(1i * (phi_obj+noise_pha));

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
sigma = 30;    % 频域高斯窗口宽度
G = exp(-((U-(u0-center)).^2 + (V-(v0-center)).^2) / (2*sigma^2));
% hanning
hann_window = zeros(N);
for i = 1:N
    for j = 1:N
        % 计算到+1级中心的距离
        dist_x = abs(j - u0);  % 列距离
        dist_y = abs(i - v0);  % 行距离
        
        % 计算汉宁窗函数值
        if dist_x <= rect_width/2 && dist_y <= rect_height/2
            % 归一化坐标到[-0.5, 0.5]
            nx = (j - u0) / (rect_width/2);
            ny = (i - v0) / (rect_height/2);
            
            % 2D汉宁窗：0.5 * (1 + cos(2*pi*r))
            % 限制在圆形区域内
            r = sqrt(nx^2 + ny^2);
            if r <= 1
                hann_window(i,j) = 0.5 * (1 + cos(pi * r));
            end
        end
    end
end
% 频域滤波
Hf = H .* Rect_1D .* 1;%32是sigima = 0

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

%% =============================
% 1. 定量评估
% =============================
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

cx = N/2 + 1;
cy = N/2 + 1;
% 选取中心矩形在 X 方向的范围
x_half = round(rect_w/(2*dx));
x1 = cx - x_half;
x2 = cx + x_half;
% 中间 / 边缘的平均相位
mean_mid  = mean(phase_x(x1:x2));
mean_edge = mean(phase_x_edge(x1:x2));

delta_phi = mean_edge - mean_mid;
fprintf('台阶边缘 vs 中间的相位高度差：Δφ = %.4f rad\n', delta_phi);

figure;
subplot(2,3,1);
surf(X*1000, Y*1000, phi_unwrap, 'EdgeColor', 'none');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('相位 (rad)');
title('原始解包裹相位','FontName','SimHei');
colormap jet; colorbar; view(30, 40); grid on;

subplot(2,3,2);% 绘制水平方向相位剖面
plot(pixel_idx, phase_x, 'b-', 'LineWidth', 2);
xlabel('X 方向像素索引', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('水平方向 (X) 相位剖面（像素域）', 'FontSize', 14, 'FontName', 'SimHei');
grid on;
xlim([1 N]);

subplot(2,3,3);% 绘制垂直方向相位剖面
plot(pixel_idx, phase_y, 'r-', 'LineWidth', 2)
xlabel('Y 位置 (mm)', 'FontSize', 12, 'FontName', 'SimHei');
ylabel('相位 (rad)', 'FontSize', 12, 'FontName', 'SimHei');
title('垂直方向 (Y) 相位剖面', 'FontSize', 14, 'FontName', 'SimHei');
grid on;
xlim([1 N])


%% =============================
% 垂直方向(Y)相位剖面 吉布斯振荡剔除 + 台阶+背景均值替代（含底部/顶部背景）
% 核心逻辑：台阶-剔除边缘振荡像素求均值；背景-剔除靠近台阶的振荡像素求均值
% =============================
% 1. 核心参数提取（与原代码一致，无需修改）
N = 512;
dy = 3.5e-6;
rect_h = 0.3e-3;   % 矩形高度
d = 0.55e-3;       % 上下矩形与中心的间距
rect_height = 100; % 频域垂直滤波器宽度（决定振荡区域）
center_idx = N/2 + 1;
phase_y_original = phase_y; % 保存原始相位剖面，用于对比
phase_y_processed = phase_y;% 初始化处理后的相位剖面

% 2. 计算吉布斯振荡区域的单侧像素数delta_N（维基数学定义/工程定义可切换）
delta_N = round(N/(2*rect_height)); % 维基定义（过冲峰值），推荐
% delta_N = round(N/rect_height);   % 工程定义（sinc零点，更大振荡区），按需切换
fprintf('垂直方向吉布斯振荡单侧剔除像素数：delta_N = %d\n', delta_N);

% 3. 计算垂直方向【3个台阶+背景区】的像素范围（行索引1~N）
h_pix = round(rect_h/(2*dy));  % 单个矩形的半高像素数
d_pix = round(d/dy);           % 上下矩形与中心的垂直间距像素数

% --- 3.1 三个台阶的像素范围（原逻辑保留）---
idx_center = (1:N) >= (center_idx - h_pix) & (1:N) <= (center_idx + h_pix); % 中心台阶
idx_up = (1:N) >= (center_idx + d_pix - h_pix) & (1:N) <= (center_idx + d_pix + h_pix); % 上台阶
idx_down = (1:N) >= (center_idx - d_pix - h_pix) & (1:N) <= (center_idx - d_pix + h_pix); % 下台阶
idx_all_steps = idx_center | idx_up | idx_down; % 所有台阶的合并范围

% --- 3.2 背景区（含底部+顶部）：排除所有台阶的区域 ---
idx_bg = ~idx_all_steps;  % 整体背景（下台阶下方=底部，上台阶上方=顶部）
% 背景区细分（可选，便于单独处理）
idx_bg_bottom = idx_bg & (1:N) < (center_idx - d_pix - h_pix); % 底部背景（下台阶下）
idx_bg_top = idx_bg & (1:N) > (center_idx + d_pix + h_pix);   % 顶部背景（上台阶上）

% 4. 台阶区处理（原逻辑完全保留，兼容空索引判断）
% --- 4.1 中心台阶 ---
idx_center_core = idx_center & (1:N) >= (center_idx - h_pix + delta_N) & (1:N) <= (center_idx + h_pix - delta_N);
if sum(idx_center_core) > 0
    avg_center = mean(phase_y(idx_center_core));
    phase_y_processed(idx_center) = avg_center;
    fprintf('中心台阶非振荡区平均相位：%.4f rad\n', avg_center);
else
    avg_center = mean(phase_y(idx_center));
    phase_y_processed(idx_center) = avg_center;
    warning('中心台阶非振荡区像素数为0，使用整个台阶均值');
end
% --- 4.2 上台阶 ---
idx_up_core = idx_up & (1:N) >= (center_idx + d_pix - h_pix + delta_N) & (1:N) <= (center_idx + d_pix + h_pix - delta_N);
if sum(idx_up_core) > 0
    avg_up = mean(phase_y(idx_up_core));
    phase_y_processed(idx_up) = avg_up;
    fprintf('上台阶非振荡区平均相位：%.4f rad\n', avg_up);
else
    avg_up = mean(phase_y(idx_up));
    phase_y_processed(idx_up) = avg_up;
    warning('上台阶非振荡区像素数为0，使用整个台阶均值');
end
% --- 4.3 下台阶 ---
idx_down_core = idx_down & (1:N) >= (center_idx - d_pix - h_pix + delta_N) & (1:N) <= (center_idx - d_pix + h_pix - delta_N);
if sum(idx_down_core) > 0
    avg_down = mean(phase_y(idx_down_core));
    phase_y_processed(idx_down) = avg_down;
    fprintf('下台阶非振荡区平均相位：%.4f rad\n', avg_down);
else
    avg_down = mean(phase_y(idx_down));
    phase_y_processed(idx_down) = avg_down;
    warning('下台阶非振荡区像素数为0，使用整个台阶均值');
end

% 5. 【新增】背景区（底部+顶部）吉布斯振荡处理（和台阶逻辑一致）
% 剔除背景区中靠近台阶边缘的delta_N个振荡像素，取核心背景区求均值
idx_bg_core = idx_bg & ...
    ( (1:N) < (center_idx - d_pix - h_pix - delta_N) | ...  % 底部背景：远离下台阶delta_N像素
      (1:N) > (center_idx + d_pix + h_pix + delta_N) );     % 顶部背景：远离上台阶delta_N像素
if sum(idx_bg_core) > 0
    avg_bg = mean(phase_y(idx_bg_core));  % 核心背景区均值（底部+顶部统一）
    phase_y_processed(idx_bg) = avg_bg;   % 均值替代整个背景区（底部+顶部）
    fprintf('背景区（底部+顶部）非振荡区平均相位：%.4f rad\n', avg_bg);
else
    avg_bg = mean(phase_y(idx_bg));       % 降级：整个背景区均值
    phase_y_processed(idx_bg) = avg_bg;
    warning('背景区非振荡区像素数为0，使用整个背景区均值');
end

% 6. 绘制对比图【新增背景区标注】，保留所有原标注
figure('Position', [100, 200, 1000, 600], 'Name', '垂直Y相位剖面-吉布斯振荡处理（含底部背景）');
plot(1:N, phase_y_original, 'b-', 'LineWidth', 1.5, 'DisplayName', '原始剖面（含吉布斯振荡）');
hold on; grid on;
plot(1:N, phase_y_processed, 'r-', 'LineWidth', 2, 'DisplayName', '处理后剖面（台阶+背景去振荡）');

% 标注：台阶中心+背景区边界
xline(center_idx, 'k--', 'LineWidth', 1, 'DisplayName', '中心轴');
xline(center_idx+d_pix, 'g--', 'LineWidth', 1, 'DisplayName', '上台阶中心');
xline(center_idx-d_pix, 'm--', 'LineWidth', 1, 'DisplayName', '下台阶中心');
xline(center_idx - d_pix - h_pix, 'c-.' , 'LineWidth', 1.2, 'DisplayName', '底部背景/下台阶分界');
xline(center_idx + d_pix + h_pix, 'y-.' , 'LineWidth', 1.2, 'DisplayName', '顶部背景/上台阶分界');

xlabel('Y方向像素索引', 'FontName', 'SimHei', 'FontSize', 12);
ylabel('相位 (rad)', 'FontName', 'SimHei', 'FontSize', 12);
title(sprintf('垂直Y相位剖面吉布斯处理（振荡剔除delta_N=%d，含底部/顶部背景）', delta_N), 'FontName', 'SimHei', 'FontSize', 14);
legend('Location', 'best', 'FontName', 'SimHei');

% 7. 输出台阶高度差（处理后，更精准，原逻辑保留）
delta_phi_up_center = avg_up - avg_center; % 上-中心台阶高度差
delta_phi_down_center = avg_down - avg_center; % 下-中心台阶高度差
delta_phi_center_bg = avg_center - avg_bg; % 中心台阶-背景高度差（新增，评估台阶与背景的相对高度）
fprintf('处理后-上台阶与中心台阶高度差：Δφ = %.4f rad\n', delta_phi_up_center);
fprintf('处理后-下台阶与中心台阶高度差：Δφ = %.4f rad\n', delta_phi_down_center);
fprintf('处理后-中心台阶与背景区高度差：Δφ = %.4f rad\n', delta_phi_center_bg);
