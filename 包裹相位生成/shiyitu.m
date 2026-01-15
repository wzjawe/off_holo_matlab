clear; clc; close all;

%% =============================
% 参数（示意用）
% =============================
N = 1024;
x = linspace(-1,1,N);
fx = linspace(-5,5,N);

%% =============================
% 左：空间域矩形窗 & 频谱泄露
% =============================
rect_win = double(abs(x) < 0.4);
sinc_resp = sinc(3*fx);

%% =============================
% 中：离轴频谱（±1 级被展宽）
% =============================
f0 = 2.5;
spec_ideal = exp(-(fx-f0).^2/0.05) ...
           + exp(-(fx+f0).^2/0.05);

spec_leak = conv(spec_ideal, sinc_resp, 'same');

%% =============================
% 右：硬截止导致 Gibbs
% =============================
hard_win = double(abs(fx-f0) < 0.6);
spec_cut = spec_leak .* hard_win;
gibbs_signal = real(ifft(ifftshift(spec_cut)));

%% =============================
% 绘图
% =============================
figure('Position',[200 200 1200 700])

% 空间域矩形窗
subplot(3,3,1)
plot(x, rect_win, 'k', 'LineWidth',1.5)
title('Spatial Rectangular Window')
xlabel('x'); ylabel('Amplitude'); grid on

% 频域 sinc
subplot(3,3,2)
plot(fx, sinc_resp, 'k', 'LineWidth',1.5)
title('Sinc in Frequency Domain')
xlabel('f_x'); ylabel('Amplitude'); grid on

% 理想频谱
subplot(3,3,4)
plot(fx, spec_ideal, 'k', 'LineWidth',1.5)
title('Ideal Off-axis Spectrum')
xlabel('f_x'); ylabel('Magnitude'); grid on

% 泄露频谱
subplot(3,3,5)
plot(fx, spec_leak, 'k', 'LineWidth',1.5)
title('Spectral Leakage (CCD Limited)')
xlabel('f_x'); ylabel('Magnitude'); grid on

% 硬截止窗
subplot(3,3,6)
plot(fx, hard_win, 'k', 'LineWidth',1.5)
title('Hard Frequency Cutoff')
xlabel('f_x'); ylabel('Window'); grid on

% Gibbs 振铃
subplot(3,3,[8 9])
plot(x, gibbs_signal/max(abs(gibbs_signal)), 'k', 'LineWidth',1.2)
title('Gibbs Artifacts in Spatial Domain')
xlabel('x'); ylabel('Normalized Amplitude'); grid on
