function [pwm, v_ilro, v_ref, v_inj] = xor_pwm(phi_bpf, p)
% XOR_PWM  相位 [rad] -> 两路逻辑载波 -> 真 XOR 逻辑波形。
% Vref 与 Vinj 正交；phi_bpf 是 Vilro 相对 Vref 的偏移，已扣除静态 pi/2。
% 慢变相位时平均占空比约 abs(phi_bpf)/pi；此式只作检查，不替代 XOR。
t = (0:numel(phi_bpf)-1)'/p.fs;
theta_inj = 2*pi*p.f_inj*t;
theta_ref = theta_inj + pi/2;
v_inj = mod(theta_inj, 2*pi) < pi;
v_ref = mod(theta_ref, 2*pi) < pi;
v_ilro = mod(theta_ref + phi_bpf(:), 2*pi) < pi;
pwm = xor(v_ilro, v_ref);
end
