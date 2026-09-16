function signals = demo_frontend(figure_dir)
% DEMO_FRONTEND  先读参数，再按五模块顺序读下面 5 行；无需音频文件/工具箱。
if nargin == 0
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    figure_dir = fullfile(root, 'reports', 'figures', 'matlab_frontend');
end
if ~isfolder(figure_dir), mkdir(figure_dir); end
p = parameters();
t = (0:round(0.04*p.fs))'/p.fs;
% 两个已知音：便于看带内/带外关系；幅度用真实 V，不按整段峰值归一化。
v_mic = 0.015*sin(2*pi*1000*t) + 0.006*sin(2*pi*3000*t);
v_aaf = aaf(v_mic, p);
v_pga = pga(v_aaf, p);
[phi_bpf, v_int] = vtc_bpf(v_pga, p);
[pwm, v_ilro, v_ref, v_inj] = xor_pwm(phi_bpf, p);
[event_time, v_iaf, v_spk] = iaf(pwm, p);
signals = struct('p',p,'t',t,'v_mic',v_mic,'v_aaf',v_aaf,'v_pga',v_pga, ...
    'phi_bpf',phi_bpf,'v_int',v_int,'pwm',pwm,'v_ilro',v_ilro, ...
    'v_ref',v_ref,'v_inj',v_inj,'event_time',event_time,'v_iaf',v_iaf,'v_spk',v_spk);

fig = figure('Color','w','Position',[80 60 1100 950]);
tiledlayout(5,1,'TileSpacing','compact');
nexttile; plot(t*1000,v_mic,t*1000,v_aaf); ylabel('V');
title('1  AAF: input voltage -> low-pass voltage'); legend('MIC','AAF');
nexttile; plot(t*1000,v_pga); ylabel('V');
title('2  PGA: fixed gain, voltage around common mode');
nexttile; plot(t*1000,phi_bpf); ylabel('rad');
title('3  VTC-BPF: signed phase relative to Vref (not output voltage)');
sl = t >= 0.030 & t <= 0.03015;
nexttile; stairs(t(sl)*1000,double(v_ref(sl))+3); hold on;
stairs(t(sl)*1000,double(v_ilro(sl))+1.5); stairs(t(sl)*1000,double(pwm(sl)));
yticks([0.5 2 3.5]); yticklabels({'XOR','Vilro','Vref'});
title('4  XOR: phase-shifted carrier edges -> PWM (150 us zoom)');
nexttile; plot(t*1000,v_iaf); hold on; yline(p.vlo,':'); yline(p.vhi,':');
plot(event_time*1000,p.vhi*ones(size(event_time)),'r.'); ylabel('V'); xlabel('Time (ms)');
title('5  IAF: PWM-gated charge/discharge, red dots = rising-edge events');
sgtitle('New single-channel MATLAB model | explicit assumptions | no silicon-fit claim');
exportgraphics(fig,fullfile(figure_dir,'five_modules.png'),'Resolution',150);

% 独立展示电压到相位的解析传函；这不是另外一次电压滤波。
f = logspace(1,4.7,600); s = 1i*2*pi*f;
H = p.G*s ./ (p.N*p.Gint*p.G + s + s.^2/p.wp);
fig = figure('Color','w','Position',[100 80 1000 500]);
tiledlayout(1,2);
nexttile; semilogx(f,abs(H),'LineWidth',1.5); xline(p.f_res,':'); grid on;
xlabel('Audio frequency (Hz)'); ylabel('|Phi / Vpga| (rad/V)');
title('Phase-domain bandpass gain');
nexttile; semilogx(f,angle(H)*180/pi,'LineWidth',1.5); grid on;
xlabel('Audio frequency (Hz)'); ylabel('Transfer phase (deg)');
title('Transfer-function phase lag (distinct from phi_bpf in rad)');
exportgraphics(fig,fullfile(figure_dir,'phase_transfer.png'),'Resolution',150);
end
