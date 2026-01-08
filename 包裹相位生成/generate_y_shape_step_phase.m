function phase = generate_y_shape_step_phase(width, height, arm_length, angle, phase_step)
% 生成Y形台阶相位图案
%
% 输入参数：
%   width, height : 相位图尺寸
%   arm_length    : Y形三条臂的长度（像素）
%   angle         : 分叉角度（度）
%   phase_step    : 相位台阶值
%
% 输出：
%   phase : 2D 解包裹前的理想台阶相位图

    if nargin < 5
        phase_step = 2*pi;
    end
    if nargin < 4
        angle = 30;
    end
    if nargin < 3
        arm_length = 100;
    end

    % -------------------------------
    % 1. 创建全零相位图
    % -------------------------------
    phase = zeros(height, width);

    % Y形中心点
    center_x = floor(width/2) + 1;
    center_y = floor(height/2) + 1;

    % 角度转弧度
    angle_rad = angle*pi/180;

    % -------------------------------
    % 2. 垂直向下的臂（第一条臂）
    % -------------------------------
    for i = 0:arm_length-1
        y = center_y + i;
        if y >= 1 && y <= height
            phase(y, center_x) = phase_step;
        end
    end

    % -------------------------------
    % 3. 左上臂
    % -------------------------------
    for i = 0:arm_length-1
        x = round(center_x - i*sin(angle_rad));
        y = round(center_y - i*cos(angle_rad));
        if x >= 1 && x <= width && y >= 1 && y <= height
            phase(y, x) = phase_step;
        end
    end

    % -------------------------------
    % 4. 右上臂
    % -------------------------------
    for i = 0:arm_length-1
        x = round(center_x + i*sin(angle_rad));
        y = round(center_y - i*cos(angle_rad));
        if x >= 1 && x <= width && y >= 1 && y <= height
            phase(y, x) = phase_step;
        end
    end

    % -------------------------------
    % 5. Y形加粗处理（对应 binary_dilation）
    % -------------------------------
    bw = phase ~= 0;

    % 使用形态学膨胀加粗
    se = strel('disk', 2);
    bw_thick = imdilate(bw, se);
    bw_thick = imdilate(bw_thick, se);
    bw_thick = imdilate(bw_thick, se);

    phase = double(bw_thick) * phase_step;

end
