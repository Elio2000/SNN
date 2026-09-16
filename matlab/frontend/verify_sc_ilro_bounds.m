function result = verify_sc_ilro_bounds(run_dir, figure_dir)
% VERIFY_SC_ILRO_BOUNDS  Audit static SC-ILRO lock limits from circuit relations.
% This is a parameter/boundary check, not a cycle-level transient simulation.
% Equations: EP4350992B1 paragraphs [0102]-[0107], consistent with
% Mostafa ISSCC 2024 Fig. 17.8.2. Unknown device values remain inferred.
assert(~isfile(fullfile(run_dir,'bounds.mat')), ...
    'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end

p = parameters();
fc = p.wp/(2*pi);
alpha = p.f_inj/(pi*fc);              % fc ~= flock/(pi*alpha), alpha = CL/CI
ci_over_cl = 1/alpha;

% Choose the free-running frequency so flock is the midpoint of the
% patent lock interval [alpha*f0/(1+alpha), f0].
f0_center = 2*p.f_inj*(1+alpha)/(1+2*alpha);
lock_low = alpha*f0_center/(1+alpha);
lock_high = f0_center;
center_relative_error = abs(mean([lock_low,lock_high])-p.f_inj)/p.f_inj;

% Vth/Vth0 = 1 + threshold_gain_ratio*DV and f0(DV) is inverse in Vth.
% The signed ratio below follows the positive phase convention in vtc_bpf.
% The patent gives negative Kinv as one circuit example; swapping Q/nQ or
% the differential input convention reverses the sign without changing limits.
threshold_gain_ratio = 2*p.G/(pi*(1+2*alpha)); % (Kinv/Vth0), 1/V
voltage_margin = pi/(2*abs(p.G));
dv = linspace(-1.2*voltage_margin,1.2*voltage_margin,241)';
vth_ratio = 1 + threshold_gain_ratio*dv;
f0 = f0_center./vth_ratio;
phase_absolute = pi*(1+(1+alpha)*(p.f_inj./f0-1));
phase_offset = phase_absolute-pi/2;
locked = phase_absolute >= 0 & phase_absolute <= pi;

edge_dv = [-voltage_margin;voltage_margin];
edge_vth_ratio = 1 + threshold_gain_ratio*edge_dv;
edge_f0 = f0_center./edge_vth_ratio;
edge_phase = pi*(1+(1+alpha)*(p.f_inj./edge_f0-1));
phase_linearity_error = max(abs(phase_offset-p.G*dv));
edge_phase_error = max(abs(edge_phase-[0;pi]));

% Patent paragraph [0107] uses the large-alpha approximation
% K2 ~= pi*alpha*Kinv/Vth0. Quantify its difference from the exact slope.
approximate_gain = pi*alpha*threshold_gain_ratio;
approximation_relative_error = abs(approximate_gain-p.G)/abs(p.G);

result = struct('parameters',p,'fc_hz',fc,'alpha_cl_over_ci',alpha, ...
    'ci_over_cl',ci_over_cl,'f0_center_hz',f0_center, ...
    'lock_low_hz',lock_low,'lock_high_hz',lock_high, ...
    'lock_width_hz',lock_high-lock_low, ...
    'center_relative_error',center_relative_error, ...
    'threshold_gain_ratio_per_v',threshold_gain_ratio, ...
    'voltage_margin_v',voltage_margin, ...
    'phase_linearity_error_rad',phase_linearity_error, ...
    'edge_phase_error_rad',edge_phase_error, ...
    'approximate_gain_rad_per_v',approximate_gain, ...
    'approximation_relative_error',approximation_relative_error);
result.pass = center_relative_error < 1e-12 && ...
    phase_linearity_error < 1e-12 && edge_phase_error < 1e-12 && ...
    ~locked(1) && ~locked(end) && approximation_relative_error < 0.01;

fprintf('MATLAB %s; %s\n',version,computer);
fprintf('ILRO low-pass fc [Hz]: %.12g\n',fc);
fprintf('Inferred alpha=CL/CI: %.12g; CI/CL: %.12g\n',alpha,ci_over_cl);
fprintf('Centered free-running f0 [Hz]: %.12g\n',f0_center);
fprintf('Static lock interval [Hz]: %.12g to %.12g; width %.12g\n', ...
    lock_low,lock_high,lock_high-lock_low);
fprintf('Threshold gain ratio |Kinv/Vth0| [1/V]: %.12g\n',abs(threshold_gain_ratio));
fprintf('Voltage margin about center [V]: +/- %.12g\n',voltage_margin);
fprintf('Phase linearity error [rad]: %.12g; edge error [rad]: %.12g\n', ...
    phase_linearity_error,edge_phase_error);
fprintf('Large-alpha gain approximation relative error: %.12g\n', ...
    approximation_relative_error);
fprintf('Static boundary checks: %d\n',result.pass);
fprintf('No cycle-level acquisition, pulling, or post-unlock waveform is modeled.\n');

save(fullfile(run_dir,'bounds.mat'),'result','dv','f0', ...
    'phase_absolute','phase_offset','locked');
summary = table(fc,alpha,ci_over_cl,f0_center,lock_low,lock_high, ...
    lock_high-lock_low,threshold_gain_ratio,voltage_margin, ...
    approximation_relative_error,result.pass, ...
    'VariableNames',{'fc_hz','alpha_cl_over_ci','ci_over_cl', ...
    'f0_center_hz','lock_low_hz','lock_high_hz','lock_width_hz', ...
    'threshold_gain_ratio_per_v','voltage_margin_v', ...
    'gain_approx_relative_error','pass'});
writetable(summary,fullfile(run_dir,'summary.csv'));

fig = figure('Visible','off','Position',[100 100 1100 760]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
nexttile;
plot(dv*1e3,phase_offset,'LineWidth',1.5); hold on;
yline(pi/2,'r--','+pi/2 lock edge');
yline(-pi/2,'r--','-pi/2 lock edge');
xline(voltage_margin*1e3,'k:'); xline(-voltage_margin*1e3,'k:');
xlabel('Differential control voltage DV [mV]');
ylabel('Phase relative to midpoint [rad]'); grid on;
title('E003 static SC-ILRO phase boundary; no transient unlock model');
nexttile;
plot(dv*1e3,f0/1e3,'LineWidth',1.5); hold on;
yline(p.f_inj/1e3,'k--','f_{lock}');
xline(voltage_margin*1e3,'r:'); xline(-voltage_margin*1e3,'r:');
xlabel('Differential control voltage DV [mV]');
ylabel('Natural frequency f_0 [kHz]'); grid on;
title(sprintf('Inferred CL/CI = %.3f; static lock width = %.1f Hz', ...
    alpha,lock_high-lock_low));
exportgraphics(fig,fullfile(figure_dir,'static_bounds.png'),'Resolution',160);
close(fig);
assert(result.pass,'Static SC-ILRO boundary audit failed.');
end
