%% =====================================================
% 最小二乘相位解包裹函数（一维版本）- 修正版、、
% =====================================================
function unwrapped_phase = least_squares_unwrap_1d_corrected(wrapped_phase, direction)
    % 一维最小二乘相位解包裹算法（修正版）
    % 输入：
    %   wrapped_phase - 包裹相位（一维向量或二维矩阵）
    %   direction - 解包裹方向
    %               1: 按列解包裹（垂直方向，默认）
    %               2: 按行解包裹（水平方向）
    % 输出：
    %   unwrapped_phase - 解包裹相位
    
    % 检查输入参数
    if nargin < 2
        direction = 1;  % 默认按列解包裹
    end
    
    % 如果输入是二维矩阵，根据方向选择解包裹方式
    if ismatrix(wrapped_phase) && size(wrapped_phase, 1) > 1 && size(wrapped_phase, 2) > 1
        unwrapped_phase = zeros(size(wrapped_phase));
        if direction == 1
            % 按列解包裹（垂直方向）
            for col = 1:size(wrapped_phase, 2)
                unwrapped_phase(:, col) = unwrap_1d_minimum_norm(wrapped_phase(:, col));
            end
        else
            % 按行解包裹（水平方向）
            for row = 1:size(wrapped_phase, 1)
                unwrapped_phase(row, :) = unwrap_1d_minimum_norm(wrapped_phase(row, :));
            end
        end
    else
        % 输入是一维向量
        if isrow(wrapped_phase)
            unwrapped_phase = unwrap_1d_minimum_norm(wrapped_phase);
        else
            unwrapped_phase = unwrap_1d_minimum_norm(wrapped_phase);
        end
    end
end

%% =====================================================
% 一维最小范数解包裹算法
% =====================================================
function phi_unwrapped = unwrap_1d_minimum_norm(phi_wrapped)
    % 一维最小范数解包裹算法
    % 参考：Herraez et al., "Fast two-dimensional phase-unwrapping algorithm
    % based on sorting by reliability following a noncontinuous path"
    
    N = length(phi_wrapped);
    
    % 计算包裹相位差
    delta_phi = zeros(N-1, 1);
    for i = 1:N-1
        delta = phi_wrapped(i+1) - phi_wrapped(i);
        % 包裹相位差，使其在[-π, π]之间
        delta_phi(i) = atan2(sin(delta), cos(delta));
    end
    
    % 构建系数矩阵A (N-1 × N)
    A = zeros(N-1, N);
    for i = 1:N-1
        A(i, i) = -1;
        A(i, i+1) = 1;
    end
    
    % 最小范数解：min ||A*φ - δ||?
    % 使用伪逆求解：φ = A? * δ
    % 其中A = A' * (A*A')
    phi_unwrapped = A' * ((A * A') \ delta_phi);
    
    % 调整常数项，使第一个点为0
    phi_unwrapped = phi_unwrapped - phi_unwrapped(1);
end

%% =====================================================
% 一维最小二乘解包裹（使用DCT方法）
% =====================================================
function phi_unwrapped = unwrap_1d_dct_method(phi_wrapped)
    % 一维DCT最小二乘解包裹算法
    % 基于二维算法的简化版本
    N = length(phi_wrapped);
    
    % 步骤1: 计算包裹相位的一阶差分（梯度）
    delta_phi = zeros(N, 1);
    for i = 1:N-1
        delta = phi_wrapped(i+1) - phi_wrapped(i);
        % 解包裹梯度
        delta_phi(i) = delta - 2*pi * round(delta / (2*pi));
    end
    
    % 步骤2: 计算ρ = ?·(?φ_wrapped)
    rho = zeros(N, 1);
    
    % 内部点
    for i = 2:N-1
        rho(i) = delta_phi(i) - delta_phi(i-1);
    end
    
    % 边界条件（Neumann边界）
    rho(1) = delta_phi(1);
    rho(N) = -delta_phi(N-1);
    
    % 步骤3: 使用DCT求解泊松方程 ??φ = ρ
    % 对ρ进行DCT变换
    rho_dct = dct(rho);
    
    % 在DCT域中求解
    phi_dct = zeros(N, 1);
    
    for k = 1:N
        if k == 1
            % 直流分量 - 设为零（解包裹相位以第一个点为参考）
            phi_dct(k) = 0;
        else
            % 计算分母：2*cos(π*(k-1)/N) - 2
            denominator = 2 * cos(pi * (k-1) / N) - 2;
            if abs(denominator) < eps
                phi_dct(k) = 0;
            else
                phi_dct(k) = rho_dct(k) / denominator;
            end
        end
    end
    
    % 步骤4: 逆DCT变换得到解包裹相位
    phi_unwrapped = idct(phi_dct);
    
    % 调整常数偏移，使第一个点与包裹相位一致
    phi_unwrapped = phi_unwrapped - phi_unwrapped(1) + phi_wrapped(1);
end

%% =====================================================
% 简单的一维相位解包裹（直接积分法）
% =====================================================
function phi_unwrapped = unwrap_1d_simple(phi_wrapped)
    % 简单的一维相位解包裹算法（直接积分法）
    % 适用于低噪声情况
    
    N = length(phi_wrapped);
    phi_unwrapped = zeros(N, 1);
    phi_unwrapped(1) = phi_wrapped(1);
    
    % 累积相位差
    for i = 2:N
        delta = phi_wrapped(i) - phi_wrapped(i-1);
        
        % 如果相位跳变超过π，进行调整
        if delta > pi
            delta = delta - 2*pi;
        elseif delta < -pi
            delta = delta + 2*pi;
        end
        
        phi_unwrapped(i) = phi_unwrapped(i-1) + delta;
    end
end

%% =====================================================
% 主解包裹函数（提供多种方法选择）
% =====================================================
function unwrapped_phase = unwrap_phase_1d(wrapped_phase, direction, method)
    % 一维相位解包裹主函数
    % 输入：
    %   wrapped_phase - 包裹相位
    %   direction - 解包裹方向（1:垂直，2:水平）
    %   method - 解包裹方法（可选）
    %              'min_norm': 最小范数法（默认）
    %              'dct': DCT最小二乘法
    %              'simple': 简单积分法
    
    if nargin < 2
        direction = 1;
    end
    if nargin < 3
        method = 'min_norm';
    end
    
    % 根据方向选择解包裹函数
    switch method
        case 'dct'
            unwrap_func = @unwrap_1d_dct_method;
        case 'simple'
            unwrap_func = @unwrap_1d_simple;
        otherwise
            unwrap_func = @unwrap_1d_minimum_norm;
    end
    
    % 处理不同维度的输入
    if ismatrix(wrapped_phase) && size(wrapped_phase, 1) > 1 && size(wrapped_phase, 2) > 1
        unwrapped_phase = zeros(size(wrapped_phase));
        if direction == 1
            % 按列解包裹
            for col = 1:size(wrapped_phase, 2)
                unwrapped_phase(:, col) = unwrap_func(wrapped_phase(:, col));
            end
        else
            % 按行解包裹
            for row = 1:size(wrapped_phase, 1)
                unwrapped_phase(row, :) = unwrap_func(wrapped_phase(row, :));
            end
        end
    else
        % 一维向量
        unwrapped_phase = unwrap_func(wrapped_phase(:));
    end
end