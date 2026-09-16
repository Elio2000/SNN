function result = verify_sampling_scope(run_dir, figure_dir)
% VERIFY_SAMPLING_SCOPE  输入保持假设的诊断；不是电容注入 ILRO 电路模型。
% 固定同一相位状态方程，只比较连续正弦输入与每周期一次的理想采样保持。
% f_update = f_inj 是待检验假设，不能据此宣称真实 ILRO 具有该采样核。
% E002 预注册：数值检查与 nominal-equivalence gate 分开报告。
assert(~isfile(fullfile(run_dir,'sampling.mat')), 'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end
timer = tic;
p = parameters(); p.fs = 1e6;
f = 1000; amplitude = 0.01; duration = 0.08;
t = (0:round(duration*p.fs))'/p.fs;
v = amplitude*sin(2*pi*f*t);
[phi, cp] = vtc_bpf(v,p);
v_hold = held_tone(numel(t),p.fs,p.f_inj,f,amplitude);
[phi_hold, cp_hold] = vtc_bpf(v_hold,p);

% 更细数值网格保持同一注入周期，不提高假设的物理更新率。
pf = p; pf.fs = 2*p.fs;
tf = (0:2*(numel(t)-1))'/pf.fs;
vf = held_tone(numel(tf),pf.fs,p.f_inj,f,amplitude);
fine = vtc_bpf(vf,pf);
refinement = max(abs(phi_hold-fine(1:2:end)))/max(abs(fine));

% 独立频域参考：理想保持器的基频系数 sinc(f/F)*exp(-j*pi*f/F)。
% 用连续传函比较，不用状态更新矩阵重新构造一个相同离散模型。
s = 1i*2*pi*f;
H = p.G*s/(p.N*p.Gint*p.G+s+s^2/p.wp);
hold_factor = sin(pi*f/p.f_inj)/(pi*f/p.f_inj)*exp(-1i*pi*f/p.f_inj);
grid_factor = sin(pi*f/p.fs)/(pi*f/p.fs)*exp(-1i*pi*f/p.fs);
h = fit_gain(t,phi,f,amplitude);
hh = fit_gain(t,phi_hold,f,amplitude);
hf = fit_gain(tf,fine,f,amplitude);
reference_error = abs(hh-H*hold_factor)/abs(H*hold_factor);
grid_reference_error = abs(h-H*grid_factor)/abs(H*grid_factor);
nominal_error = abs(hh-H)/abs(H);
result = struct('parameters',p,'frequency_hz',f,'amplitude_v',amplitude, ...
    'duration_s',duration,'continuous_gain',H,'grid_gain',h, ...
    'held_gain',hh,'fine_held_gain',hf,'ideal_hold_gain',H*hold_factor, ...
    'hold_reference_relative_error',reference_error, ...
    'grid_reference_relative_error',grid_reference_error, ...
    'time_step_relative_error',refinement, ...
    'nominal_complex_relative_error',nominal_error, ...
    'nominal_amplitude_relative_error',abs(abs(hh)/abs(H)-1), ...
    'nominal_phase_error_deg',angle(hh/H)*180/pi, ...
    'max_abs_phase_rad',max(abs(phi_hold)), ...
    'max_abs_cp_v',max(abs(cp_hold)));
result.numerical_pass = reference_error < 2e-4 && ...
    grid_reference_error < 2e-4 && refinement < 1e-8;
result.equivalence_pass = nominal_error < 0.02;
result.wall_seconds = toc(timer);
fprintf('MATLAB %s; %s\n',version,computer);
fprintf('Hold reference relative error: %.12g\n',reference_error);
fprintf('Grid reference relative error: %.12g\n',grid_reference_error);
fprintf('Time-step relative error: %.12g\n',refinement);
fprintf('Nominal complex relative error: %.12g\n',nominal_error);
fprintf('Nominal amplitude relative error: %.12g\n',result.nominal_amplitude_relative_error);
fprintf('Nominal phase error [deg]: %.12g\n',result.nominal_phase_error_deg);
fprintf('Max absolute phase [rad]: %.12g; CP state [V]: %.12g\n', ...
    result.max_abs_phase_rad,result.max_abs_cp_v);
fprintf('Numerical checks: %d; nominal equivalence gate: %d\n', ...
    result.numerical_pass,result.equivalence_pass);
fprintf('These results do not test physical injection locking or capture.\n');
save(fullfile(run_dir,'sampling.mat'),'result','t','v','v_hold', ...
    'phi','cp','phi_hold','cp_hold');
summary = table([p.fs;p.fs;pf.fs],[p.fs;p.f_inj;p.f_inj], ...
    abs([h;hh;hf]),angle([h;hh;hf])*180/pi,abs([h;hh;hf]-H)/abs(H), ...
    'VariableNames',{'grid_hz','update_hz','gain_rad_per_v','phase_deg','relative_error'});
writetable(summary,fullfile(run_dir,'summary.csv'));

fig = figure('Visible','off','Position',[100 100 1100 820]);
tiledlayout(3,1,'Padding','compact','TileSpacing','compact');
nexttile; keep = t >= .040 & t <= .04025;
plot((t(keep)-.04)*1e6,v(keep)*1e3,'LineWidth',1.4); hold on;
stairs((t(keep)-.04)*1e6,v_hold(keep)*1e3,'LineWidth',1.2);
xlabel('Time after 40 ms [us]'); ylabel('PGA input [mV]'); grid on;
legend('1 MHz numerical input','50 kHz ideal hold','Location','northwest');
title('E002: input-hold sensitivity; not a physical injection model');
nexttile; keep = t >= .040 & t <= .042;
plot((t(keep)-.04)*1e3,phi(keep),'LineWidth',1.4); hold on;
plot((t(keep)-.04)*1e3,phi_hold(keep),'--','LineWidth',1.4);
xlabel('Time after 40 ms [ms]'); ylabel('Signed phase [rad]'); grid on;
legend('Average input','50 kHz hold','Location','best');
nexttile;
bar([100*result.nominal_amplitude_relative_error, ...
    100*nominal_error]); hold on; yline(2,'r--','2% complex gate');
xticks(1:2); xticklabels({'Amplitude error','Complex gain error'});
ylabel('Error relative to continuous transfer [%]'); grid on;
title(sprintf('50 kHz update, 1 kHz tone: phase shift %.3f deg',result.nominal_phase_error_deg));
exportgraphics(fig,fullfile(figure_dir,'sampling_scope.png'),'Resolution',160);
close(fig);
assert(result.numerical_pass,'Numerical checks failed; do not interpret the equivalence gate.');
end

function v = held_tone(n,fs,rate,f,amplitude)
ratio = fs/rate;
assert(ratio == round(ratio),'Update periods must align with this diagnostic grid.');
sample_time = floor((0:n-1)'/ratio)/rate;
v = amplitude*sin(2*pi*f*sample_time);
end

function h = fit_gain(t,phi,f,amplitude)
keep = t >= .04 & t < .08; % 40 整周期；排除终点的重复相位样本
basis = [sin(2*pi*f*t(keep)),cos(2*pi*f*t(keep)),ones(nnz(keep),1)];
fit = basis\phi(keep);
h = (fit(1)+1i*fit(2))/amplitude;
end
