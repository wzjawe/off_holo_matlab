%% =====================================================
% 最小二乘相位解包裹函数
% =====================================================
function phs = least_squares_unwrap(wrapped_phs)
    % 最小二乘相位解包裹算法
    % 输入：wrapped_phs - 包裹相位图
    % 输出：phs - 解包裹相位
    
    a = wrapped_phs;            % 将包裹相位赋值给 a
    [M, N] = size(a);           % 计算二维包裹相位的大小(行、列数)
    
    % 预设包裹相位沿 x 方向和 y 方向的梯度
    dx = zeros(M, N);
    dy = zeros(M, N);
    
    % 计算包裹相位沿 x 方向的梯度
    for m = 1:M-1
        dx(m, :) = a(m+1, :) - a(m, :);
    end
    dx = dx - pi * round(dx / pi); % 去除梯度中的跳跃
    
    % 计算包裹相位沿 y 方向的梯度
    for n = 1:N-1
        dy(:, n) = a(:, n+1) - a(:, n);
    end
    dy = dy - pi * round(dy / pi); % 去除梯度中的跳跃
    
    % 为计算 ρ_nm 作准备
    p = zeros(M, N);
    p1 = zeros(M, N);
    p2 = zeros(M, N);
    
    % 计算 Δgx_nm - Δgx_(n-1)m
    for m = 2:M
        p1(m, :) = dx(m, :) - dx(m-1, :);
    end
    
    % 计算 Δgy_nm - Δgy_n(m-1)
    for n = 2:N
        p2(:, n) = dy(:, n) - dy(:, n-1);
    end
    
    p = p1 + p2;                   % 计算 ρ_nm
    
    % 边界处理
    p(1, 1) = dx(1, 1) + dy(1, 1);
    for n = 2:N
        p(1, n) = dx(1, n) + dy(1, n) - dy(1, n-1);
    end
    for m = 2:M
        p(m, 1) = dx(m, 1) - dx(m-1, 1) + dy(m, 1);
    end
    
    % DCT 变换
    pp = dct2(p) + eps;            % 对 ρ_nm 进行二维离散余弦变换
    
    % 计算 φ_nm 在 DCT 域的精确解
    fi = zeros(M, N);
    for m = 1:M
        for n = 1:N
            denominator = 2 * cos(pi * (m-1) / M) + 2 * cos(pi * (n-1) / N) - 4 + eps;
            fi(m, n) = pp(m, n) / denominator;
        end
    end
    
    fi(1, 1) = pp(1, 1);           % 赋值 DCT 域的 Φ_11
    
    % 用 iDCT 计算解包裹相位在空域中的值
    phs = idct2(fi);               % 二维逆离散余弦变换
end