%%
clear; clc; close all;

N = 100;
x = (1:N)';

% ---------- 理想阶跃 ----------
step = zeros(N,1);
step(x > N/2) = 1;

% ---------- 频谱截断（制造 Gibbs） ----------
F = fft(step);

cutoff = round(0.08*N);   % 截断比例，越小 Gibbs 越强
H = zeros(N,1);
H(1:cutoff) = 1;
H(end-cutoff+1:end) = 1;

step_gibbs = real(ifft(F .* H));

% ---------- 对比 ----------
figure;
plot(step,'k--','LineWidth',1.2); hold on;
plot(step_gibbs,'r','LineWidth',1.5);
legend('理想阶跃','Gibbs 振铃');
title('1D Gibbs 振铃的正确构造','FontSize', 14, 'FontName', 'SimHei');
grid on;

M  = 8;     % 亚像素分辨率（论文 4~8）
K1 = 2;     % TV 起点
K2 = 6;     % TV 终点

C0 = fft(step_gibbs);
Is = zeros(N, 2*M);

x0 = (0:N-1)';

for s = -M:(M-1)
    shift = s/(2*M);
    phase_ramp = exp(-1i*2*pi*x0*shift/N);
    Is(:, s+M+1) = real(ifft(C0 .* phase_ramp));
end
figure;
plot(Is(:,M+1),'k'); hold on;
plot(Is(:,M+2),'r--');
title('相邻亚像素 shift 信号（应非常接近但不相同）','FontSize', 14, 'FontName', 'SimHei');


ix = N/2 + 8;   % 边缘右侧 2~5 像素最典型

TVp = zeros(2*M,1);
TVm = zeros(2*M,1);

for si = 1:2*M
    tmp = Is(:,si);

    for n = K1:K2
        TVp(si) = TVp(si) + abs(tmp(ix+n) - tmp(ix+n-1));
        TVm(si) = TVm(si) + abs(tmp(ix-n) - tmp(ix-n+1));
    end
end

figure;
plot(TVp,'r-o','LineWidth',1.2); hold on;
plot(TVm,'b-s','LineWidth',1.2);
legend('TV^+','TV^-');
title('不同亚像素 shift 的 TV 能量','FontSize', 14, 'FontName', 'SimHei');
grid on;

%%选择最优TV
[minp, ip] = min(TVp);
[minm, im] = min(TVm);

if minp < minm
    r = ip - (M+1);
    sel = ip;
else
    r = im - (M+1);
    sel = im;
end

fprintf('最优亚像素 shift = %d / (2M) pixel\n', r);

xq = ix - r/(2*M);
val_unring = interp1(1:N, Is(:,sel), xq, 'linear');

step_unring = step_gibbs;

for ix = (K2+2):(N-K2-1)

    TVp = zeros(2*M,1);
    TVm = zeros(2*M,1);

    for si = 1:2*M
        tmp = Is(:,si);

        for n = K1:K2
            TVp(si) = TVp(si) + abs(tmp(ix+n) - tmp(ix+n-1));
            TVm(si) = TVm(si) + abs(tmp(ix-n) - tmp(ix-n+1));
        end
    end

    [minp, ip] = min(TVp);
    [minm, im] = min(TVm);

    if minp < minm
        r = ip - (M+1);
        sel = ip;
    else
        r = im - (M+1);
        sel = im;
    end

    xq = ix - r/(2*M);
    step_unring(ix) = interp1(1:N, Is(:,sel), xq, 'linear');
end


figure;
plot(step_gibbs,'k','LineWidth',1.2); hold on;
plot(step_unring,'r','LineWidth',1.8);
legend('Gibbs','Subvoxel + TV');
grid on;
xlim([N/2-40, N/2+40]);
ylim([0.9 1.1]);
title('1D Gibbs 去振铃效果（放大边缘）','FontSize', 14, 'FontName', 'SimHei');


figure;
stem(-M:(M-1), TVp, 'filled');
xlabel('r');
ylabel('TV^+');
grid on;

