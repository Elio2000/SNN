function result = verify_sc_ilro_cycle_map(run_dir, figure_dir)
% VERIFY_SC_ILRO_CYCLE_MAP  Test local acquisition of the normalized cycle map.
% The map is defined only on the published locked branch. It deliberately
% returns invalid outside that branch instead of inventing pulling dynamics.
assert(~isfile(fullfile(run_dir,'cycle_map.mat')), ...
    'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end

p = parameters();
fc = p.wp/(2*pi);
alpha = p.f_inj/(pi*fc);
f0_center = 2*p.f_inj*(1+alpha)/(1+2*alpha);
threshold_gain_ratio = 2*p.G/(pi*(1+2*alpha));
voltage_margin = pi/(2*abs(p.G));

fractions = [-0.8,-0.4,0,0.4,0.8];
dv = fractions*voltage_margin;
f0 = f0_center./(1+threshold_gain_ratio*dv);
expected = pi*(1+(1+alpha)*(p.f_inj./f0-1));
n_cycles = 600;
phase = zeros(n_cycles+1,numel(fractions));
phase(1,:) = pi/2;
branch_valid = true(n_cycles,numel(fractions));
for k = 1:n_cycles
    for j = 1:numel(fractions)
        [phase(k+1,j),branch_valid(k,j)] = ...
            sc_ilro_cycle_map(phase(k,j),f0(j),p.f_inj,alpha);
    end
end
final_error = abs(phase(end,:)-expected);
fixed_residual = zeros(size(expected));
for j = 1:numel(expected)
    fixed_residual(j) = abs(sc_ilro_cycle_map(expected(j), ...
        f0(j),p.f_inj,alpha)-expected(j));
end

settle_cycles = zeros(size(expected));
for j = 1:numel(expected)
    hit = find(abs(phase(:,j)-expected(j)) < 1e-3,1)-1;
    if isempty(hit), hit = NaN; end
    settle_cycles(j) = hit;
end

% Local small-signal pole at the center. Convert per-cycle contraction to Hz.
eps_phi = 1e-6;
plus = sc_ilro_cycle_map(pi/2+eps_phi,f0_center,p.f_inj,alpha);
minus = sc_ilro_cycle_map(pi/2-eps_phi,f0_center,p.f_inj,alpha);
contraction = (plus-minus)/(2*eps_phi);
fc_map = -log(abs(contraction))*p.f_inj/(2*pi);
fc_relative_error = abs(fc_map-fc)/fc;

% Causality: two f0 sequences with a common 100-cycle prefix must produce
% identical phase on that prefix, even when their future suffixes differ.
prefix = 100; total = 200;
seq_a = f0_center*ones(total,1); seq_b = seq_a;
seq_a(prefix+1:end) = f0(2); seq_b(prefix+1:end) = f0(4);
path_a = zeros(total+1,1); path_b = path_a;
path_a(1) = pi/2; path_b(1) = pi/2;
for k = 1:total
    path_a(k+1) = sc_ilro_cycle_map(path_a(k),seq_a(k),p.f_inj,alpha);
    path_b(k+1) = sc_ilro_cycle_map(path_b(k),seq_b(k),p.f_inj,alpha);
end
causal_prefix_exact = isequal(path_a(1:prefix+1),path_b(1:prefix+1));

outside_dv = [-1.05,1.05]*voltage_margin;
outside_f0 = f0_center./(1+threshold_gain_ratio*outside_dv);
outside_phase = pi*(1+(1+alpha)*(p.f_inj./outside_f0-1));
outside_rejected = all(outside_phase < 0 | outside_phase > pi);

result = struct('parameters',p,'alpha_cl_over_ci',alpha, ...
    'fc_target_hz',fc,'fc_map_hz',fc_map, ...
    'fc_relative_error',fc_relative_error,'contraction_per_cycle',contraction, ...
    'fractions_of_voltage_margin',fractions,'dv_v',dv,'f0_hz',f0, ...
    'expected_phase_rad',expected,'final_error_rad',final_error, ...
    'fixed_residual_rad',fixed_residual,'settle_cycles_1mrad',settle_cycles, ...
    'all_cycles_in_branch',all(branch_valid,'all'), ...
    'causal_prefix_exact',causal_prefix_exact, ...
    'outside_dv_v',outside_dv,'outside_phase_rad',outside_phase, ...
    'outside_rejected',outside_rejected);
result.pass = max(final_error) < 1e-6 && max(fixed_residual) < 1e-12 && ...
    result.all_cycles_in_branch && causal_prefix_exact && ...
    fc_relative_error < 0.02 && outside_rejected;

fprintf('MATLAB %s; %s\n',version,computer);
fprintf('alpha=CL/CI: %.12g; injection cycles: %d\n',alpha,n_cycles);
fprintf('Target phases [rad]:'); fprintf(' %.9g',expected); fprintf('\n');
fprintf('Final errors [rad]:'); fprintf(' %.9g',final_error); fprintf('\n');
fprintf('Settle cycles to 1 mrad:'); fprintf(' %g',settle_cycles); fprintf('\n');
fprintf('Fixed-point max residual [rad]: %.12g\n',max(fixed_residual));
fprintf('Center contraction/cycle: %.12g\n',contraction);
fprintf('Map fc [Hz]: %.12g; target fc [Hz]: %.12g; relative error %.12g\n', ...
    fc_map,fc,fc_relative_error);
fprintf('All cycles in branch: %d; causal prefix exact: %d\n', ...
    result.all_cycles_in_branch,causal_prefix_exact);
fprintf('Outside phases [rad]: %.9g %.9g; rejected: %d\n', ...
    outside_phase(1),outside_phase(2),outside_rejected);
fprintf('Normalized locked-branch cycle-map checks: %d\n',result.pass);
fprintf('No pulling, cycle-slip, or transistor transient is generated.\n');

save(fullfile(run_dir,'cycle_map.mat'),'result','phase','branch_valid', ...
    'path_a','path_b','seq_a','seq_b');
summary = table(fractions(:),dv(:),f0(:),expected(:),phase(end,:)', ...
    final_error(:),settle_cycles(:), ...
    'VariableNames',{'margin_fraction','dv_v','f0_hz','expected_phase_rad', ...
    'final_phase_rad','final_error_rad','settle_cycles_1mrad'});
writetable(summary,fullfile(run_dir,'summary.csv'));

cycles = (0:n_cycles)';
fig = figure('Visible','off','Position',[100 100 1150 880]);
tiledlayout(3,1,'Padding','compact','TileSpacing','compact');
nexttile;
plot(cycles,phase,'LineWidth',1.25); grid on;
xlabel('Injection cycle'); ylabel('Absolute phase Phi [rad]');
legend(compose('DV = %.1f mV',dv*1e3),'Location','eastoutside');
title('E004 normalized SC-ILRO local acquisition inside 0 <= Phi <= pi');
nexttile;
semilogy(cycles,max(abs(phase-expected),1e-16),'LineWidth',1.25); grid on;
yline(1e-6,'r--','final gate');
xlabel('Injection cycle'); ylabel('|Phi - Phi_*| [rad]');
title(sprintf('Center equivalent fc %.2f Hz; target %.2f Hz',fc_map,fc));
nexttile;
plot(dv*1e3,expected,'o-','LineWidth',1.25); hold on;
plot(outside_dv*1e3,outside_phase,'rx','MarkerSize',9,'LineWidth',1.5);
yline(0,'r--'); yline(pi,'r--'); grid on;
xlabel('Differential control voltage DV [mV]'); ylabel('Fixed phase Phi_* [rad]');
title('Red crosses are rejected; no out-of-branch waveform is synthesized');
exportgraphics(fig,fullfile(figure_dir,'cycle_map.png'),'Resolution',160);
close(fig);
assert(result.pass,'Normalized SC-ILRO cycle-map audit failed.');
end
