%% =====================================================
%  1D FFT 离轴全息相位重建 + 频域带宽扫描分析
%  (融合 test1 平台误差分析思想)
% =====================================================
clear; clc; close all;

%% =============================
% 1. 参数设置
% =============================
N = 512;
dx = 3.5e-6;
lambda = 632.8e-9;
k = 2*pi/lambda;

x = (-N/2:N/2-1)*dx;
[X,Y] = meshgrid(x,x);

%% =============================
% 2. 构造五矩形台阶相位
% =============================
phi_obj = zeros(N);

phi_c = 0.2*pi;
phi_u = 0.6*pi;
phi_d = 0.5*pi;
phi_l = 0.3*pi;
phi_r = 0.4*pi;

rect_w = 0.3e-3;
rect_h = 0.3e-3;
d = 0.55e-3;

phi_obj(abs(X)<=rect_w/2 & abs(Y)<=rect_h/2) = phi_c;
phi_obj(abs(X)<=rect_w/2 & abs(Y-d)<=rect_h/2) = phi_u;
phi_obj(abs(X)<=rect_w/2 & abs(Y+d)<=rect_h/2) = phi_d;
phi_obj(abs(X+d)<=rect_w/2 & abs(Y)<=rect_h/2) = phi_l;
phi_obj(abs(X-d)<=rect_w/2 & abs(Y)<=rect_h/2) = phi_r;

O = exp(1i*phi_obj);

%% =============================
% 3. 离轴参考光
% =============================
theta_y = 1.5*pi/180;
R = exp(1i*k*sin(theta_y)*Y);

%% =============================
% 4. 全息图 + 1D FFT
% =============================
I = abs(O + R).^2;
H = fftshift(fft(I,[],1),1);
H_amp = abs(H);

%% =============================
% 5. +1级自动检测
% =============================
center = N/2+1;
spec_v = mean(abs(H),2);
spec_v(center-30:center+30) = 0;
[~, v0] = max(spec_v);
u0 = center;

fprintf('检测到 +1 级位置：u0=%d, v0=%d\n',u0,v0);

%% =============================
% 6. 频域带宽扫描
% =============================
rect_width = N;
rect_height_list = 40:20:260;

avg_phase_list = zeros(size(rect_height_list));
rmse_list = zeros(size(rect_height_list));

u = (1:N)-center;
v = (1:N)-center;
[U,V] = meshgrid(u,v);

u_c = u0-center;
v_c = v0-center;

for ii = 1:length(rect_height_list)

    rect_height = rect_height_list(ii);

    Rect = zeros(N);
    Rect(abs(U-u_c)<=rect_width/2 & ...
         abs(V-v_c)<=rect_height/2) = 1;

    Hf = H .* Rect;

    % 中心化
    shift_x = center-u0;
    shift_y = center-v0;
    Hf_center = circshift(Hf,[shift_y shift_x]);

    % IFFT
    U_rec = ifft(Hf_center,[],1);
    phi_wrapped = angle(U_rec);
    phi_unwrap = unwrap_1d_ls(phi_wrapped);

    % 平台区域（避开边缘）
    cx = N/2+1;
    cy = N/2+1;
    x_half = round(rect_w/(2*dx));
    y_half = round(rect_h/(2*dx));

    region = phi_unwrap(cy-y_half+10:cy+y_half-10,...
                        cx-x_half+10:cx+x_half-10);

    avg_phase_list(ii) = mean(region(:));
    rmse_list(ii) = sqrt(mean((phi_unwrap(:)-phi_obj(:)).^2));
end

%% =============================
% 7. 结果可视化
% =============================
figure;

subplot(1,2,1);
plot(rect_height_list,avg_phase_list,'o-','LineWidth',2);
xlabel('频域窗口高度');
ylabel('平台平均相位 (rad)');
title('平台高度 vs 频域带宽');
grid on;

subplot(1,2,2);
plot(rect_height_list,rmse_list,'s-','LineWidth',2);
xlabel('频域窗口高度');
ylabel('RMSE (rad)');
title('重建误差 vs 频域带宽');
grid on;

%% =============================
% 8. 最佳结果展示
% =============================
[~,idx_best] = min(rmse_list);
best_height = rect_height_list(idx_best);
fprintf('最优窗口高度 = %d 像素\n',best_height);

%% =============================
% ====== 1D 最小二乘解包裹函数 ======
% =============================
function phi_out = unwrap_1d_ls(phi_in)

[Nx,Ny] = size(phi_in);
phi_out = zeros(size(phi_in));

for j = 1:Ny
    p = phi_in(:,j);
    dp = diff(p);
    dp = mod(dp+pi,2*pi)-pi;
    dp = [dp(1); dp];
    phi_out(:,j) = cumsum(dp);
end

end