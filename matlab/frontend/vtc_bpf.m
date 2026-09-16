function [phi_bpf, v_int] = vtc_bpf(v_pga, p)
% VTC_BPF  输入电压 [V] -> 相对 Vref 的有符号带通相位 [rad]。
% 论文 Fig.17.8.2/3 的锁定态小信号模型；不是电压 BPF 后改名。
% 两个物理状态：phi [rad]；v_int [V]（CP 的正号积分状态，求和端取负）。
%   dphi/dt = wp * (G*(v_pga - N*v_int) - phi)
%   dv_int/dt = Gint * phi     实际注入求和端为 -N*v_int
% 所以 Phi/Vpga = G*s / (N*Gint*G + s + s^2/wp)，与原图一致。
% 此层是平均相位动态，未包含逐注入周期采样、非线性失锁和晶体管噪声。
A = [-p.wp, -p.wp*p.G*p.N; p.Gint, 0];
B = [p.wp*p.G; 0];
M = expm([A, B; 0, 0, 0]/p.fs); % 网格内电压保持，精确更新线性状态
x = [0; 0];
phi_bpf = zeros(size(v_pga));
v_int = zeros(size(v_pga));
for k = 1:numel(v_pga)
    phi_bpf(k) = x(1);
    v_int(k) = x(2);
    x = M(1:2,1:2)*x + M(1:2,3)*v_pga(k);
end
assert(all(abs(phi_bpf) < pi/2), ...
    'Phase exceeds locked small-signal model range; do not clip it to fake locking.');
end
