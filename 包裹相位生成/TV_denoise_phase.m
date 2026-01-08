function phs = TV_denoise_phase(phs_in, lambda)

    % 使用梯度下降的简化 TV 相位去噪
    phs = phs_in;

    for iter = 1:30

        % 计算梯度
        dx = diff(phs,1,2);
        dy = diff(phs,1,1);

        dx(:,end+1) = 0;
        dy(end+1,:) = 0;

        grad = [dx(:,1:end-1)-dx(:,2:end), zeros(size(phs,1),1)] + ...
               [dy(1:end-1,:)-dy(2:end,:); zeros(1,size(phs,2))];

        % 更新
        phs = phs - lambda * grad;

        % 与原始包裹相位保持一致（模2π意义）
        phs = angle(exp(1i * phs));
    end
end
