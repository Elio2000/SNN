function v_pga = pga(v_aaf, p)
% PGA  输入 AAF 电压 [V]，输出放大小信号电压 [V]。
% 物理工作点是 p.vcm；此处只计算围绕共模的差分变化，不仿真 CMFB。
% 固定增益；不包含 AGC、逐音频归一化或虚构的饱和补偿。
assert(p.gain_db >= 0 && p.gain_db <= 12, 'PGA gain must be 0..12 dB.');
v_pga = 10^(p.gain_db/20)*v_aaf;
end
