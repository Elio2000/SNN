function v_aaf = aaf(v_mic, p)
% AAF  输入/输出均为小信号电压 [V]。论文 Fig.17.8.1：二阶低通。
% x1 = 输出电压 [V]，x2 = 电压变化率 [V/s]。
% H(s) = wc^2 / (s^2 + wc/q*s + wc^2)。阻尼、截止是显式模型假设。
w = 2*pi*p.f_aaf;
A = [0, 1; -w^2, -w/p.q_aaf];
B = [0; w^2];
% 每一网格间隔将输入保持不变，精确积分线性状态；只依赖基础 MATLAB。
M = expm([A, B; 0, 0, 0]/p.fs);
x = [0; 0];
v_aaf = zeros(size(v_mic));
for k = 1:numel(v_mic)
    v_aaf(k) = x(1);
    x = M(1:2,1:2)*x + M(1:2,3)*v_mic(k);
end
end
