function result = verify_cascade_range(run_dir, figure_dir)
% VERIFY_CASCADE_RANGE  Dynamic XOR/IAF and full-chain amplitude-range audit.
% The only swept signal variable is 1-kHz microphone amplitude. A 2x grid
% check is numerical validation, not a second scientific treatment.
assert(~isfile(fullfile(run_dir,'cascade.mat')), ...
    'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end

p = parameters();
f = 1000; duration = 0.03;
w = 2*pi*f; wc = 2*pi*p.f_aaf;
h_aaf = wc^2/((1i*w)^2+(wc/p.q_aaf)*(1i*w)+wc^2);
pga_gain = 10^(p.gain_db/20);
mic_limit = (pi/2)/(p.G*abs(h_aaf)*pga_gain);
fractions = [0,0.05,0.1,0.2,0.4,0.6,0.8,0.95,1.05];
amplitudes = fractions*mic_limit;

n = numel(fractions);
accepted = false(1,n); phase_amplitude = nan(1,n);
duty = nan(1,n); duty_reference = nan(1,n); window_duty_mae = nan(1,n);
event_count = nan(1,n); analytic_event_count = nan(1,n);
selected = struct();
for j = 1:n
    try
        [m,s] = chain_metrics(amplitudes(j),p,f,duration);
        accepted(j) = true;
        phase_amplitude(j) = m.phase_amplitude;
        duty(j) = m.duty;
        duty_reference(j) = m.duty_reference;
        window_duty_mae(j) = m.window_duty_mae;
        event_count(j) = m.event_count;
        analytic_event_count(j) = m.analytic_event_count;
        if fractions(j) == 0.4, selected = s; end
    catch err
        if ~contains(err.message,'Phase exceeds'), rethrow(err); end
    end
end

valid = fractions <= 0.95;
invalid = fractions >= 1.05;
range_pass = all(accepted(valid)) && all(~accepted(invalid));
aggregate_duty_error = abs(duty-duty_reference);
dynamic_pwm_pass = max(aggregate_duty_error(valid)) <= 2*p.f_inj/p.fs+1e-9 && ...
    max(window_duty_mae(valid)) <= 2*p.f_inj/p.fs+0.002;
iaf_error = abs(event_count-analytic_event_count);
iaf_pass = max(iaf_error(valid)) <= 1 && ...
    all(diff(event_count(valid)) >= 0) && event_count(1) == 0;

% Equal-magnitude opposite-sign input must retain full-wave statistics.
[negative,~] = chain_metrics(-0.4*mic_limit,p,f,duration);
positive_index = find(fractions == 0.4,1);
sign_duty_error = abs(negative.duty-duty(positive_index));
sign_event_error = abs(negative.event_count-event_count(positive_index));
sign_pass = sign_duty_error <= 0.002 && sign_event_error <= 1;

% Numerical refinement at one fixed physical amplitude.
p2 = p; p2.fs = 2*p.fs;
[fine,~] = chain_metrics(0.4*mic_limit,p2,f,duration);
refine_phase_error = abs(fine.phase_amplitude/phase_amplitude(positive_index)-1);
refine_duty_error = abs(fine.duty-duty(positive_index));
refine_event_error = abs(fine.event_count-event_count(positive_index));
refinement_pass = refine_phase_error < 0.005 && refine_duty_error <= 0.002 && ...
    refine_event_error <= 1;

result = struct('parameters',p,'tone_hz',f,'duration_s',duration, ...
    'predicted_mic_limit_vpeak',mic_limit,'fractions',fractions, ...
    'amplitudes_vpeak',amplitudes,'accepted',accepted, ...
    'phase_amplitude_rad',phase_amplitude,'duty',duty, ...
    'duty_reference',duty_reference,'aggregate_duty_error',aggregate_duty_error, ...
    'window_duty_mae',window_duty_mae,'event_count',event_count, ...
    'analytic_event_count',analytic_event_count,'iaf_count_error',iaf_error, ...
    'sign_duty_error',sign_duty_error,'sign_event_error',sign_event_error, ...
    'refine_phase_error',refine_phase_error, ...
    'refine_duty_error',refine_duty_error, ...
    'refine_event_error',refine_event_error, ...
    'range_pass',range_pass,'dynamic_pwm_pass',dynamic_pwm_pass, ...
    'iaf_pass',iaf_pass,'sign_pass',sign_pass, ...
    'refinement_pass',refinement_pass);
result.pass = range_pass && dynamic_pwm_pass && iaf_pass && sign_pass && refinement_pass;

fprintf('MATLAB %s; %s\n',version,computer);
fprintf('Predicted 1-kHz microphone lock limit [Vpeak]: %.12g\n',mic_limit);
fprintf('Accepted:'); fprintf(' %d',accepted); fprintf('\n');
fprintf('Phase amplitudes [rad]:'); fprintf(' %.9g',phase_amplitude); fprintf('\n');
fprintf('Aggregate duty errors:'); fprintf(' %.9g',aggregate_duty_error); fprintf('\n');
fprintf('Window duty MAE:'); fprintf(' %.9g',window_duty_mae); fprintf('\n');
fprintf('IAF counts:'); fprintf(' %g',event_count); fprintf('\n');
fprintf('IAF analytic counts:'); fprintf(' %g',analytic_event_count); fprintf('\n');
fprintf('Sign duty/count errors: %.12g / %.12g\n',sign_duty_error,sign_event_error);
fprintf('2x-grid phase/duty/count errors: %.12g / %.12g / %.12g\n', ...
    refine_phase_error,refine_duty_error,refine_event_error);
fprintf('Range/PWM/IAF/sign/refinement checks: %d %d %d %d %d\n', ...
    range_pass,dynamic_pwm_pass,iaf_pass,sign_pass,refinement_pass);
fprintf('Dynamic PWM and cascade-range checks: %d\n',result.pass);
fprintf('The range excludes unmodeled PGA saturation and physical lock transients.\n');

save(fullfile(run_dir,'cascade.mat'),'result','selected');
summary = table(fractions(:),amplitudes(:),accepted(:),phase_amplitude(:), ...
    duty(:),duty_reference(:),aggregate_duty_error(:),window_duty_mae(:), ...
    event_count(:),analytic_event_count(:),iaf_error(:), ...
    'VariableNames',{'limit_fraction','mic_vpeak','accepted', ...
    'phase_amplitude_rad','duty','duty_reference','duty_error', ...
    'window_duty_mae','event_count','analytic_event_count','event_count_error'});
writetable(summary,fullfile(run_dir,'summary.csv'));

fig = figure('Visible','off','Position',[100 100 1150 880]);
tiledlayout(3,1,'Padding','compact','TileSpacing','compact');
nexttile;
plot(fractions(valid),phase_amplitude(valid),'o-','LineWidth',1.3); hold on;
yline(pi/2,'r--','phase lock edge');
plot(fractions(invalid),pi/2*ones(nnz(invalid),1),'rx','MarkerSize',9,'LineWidth',1.5);
xlabel('Input amplitude / predicted lock limit'); ylabel('Phase amplitude [rad]'); grid on;
title('E005 full-chain locked range; red cross is rejected');
nexttile;
plot(fractions(valid),duty(valid),'o-','LineWidth',1.3); hold on;
plot(fractions(valid),duty_reference(valid),'x--','LineWidth',1.2);
xlabel('Input amplitude / predicted lock limit'); ylabel('Mean PWM duty'); grid on;
legend('XOR edges','mean(|Phi|)/pi','Location','northwest');
title(sprintf('Dynamic PWM at %.0f samples/carrier period',p.fs/p.f_inj));
nexttile;
stairs((selected.t-0.01)*1e3,selected.phi,'LineWidth',1.1); hold on;
yyaxis right; stairs((selected.t-0.01)*1e3,double(selected.pwm),'LineWidth',0.8);
xlim([0,2]); xlabel('Time after 10 ms [ms]'); grid on;
yyaxis left; ylabel('Phase [rad]'); yyaxis right; ylabel('PWM logic');
title('0.4x-limit example: dynamic phase to actual XOR waveform');
exportgraphics(fig,fullfile(figure_dir,'dynamic_cascade.png'),'Resolution',160);
close(fig);
assert(result.pass,'Dynamic PWM/cascade range audit failed.');
end

function [m,s] = chain_metrics(amplitude,p,f,duration)
t = (0:round(duration*p.fs))'/p.fs;
v_mic = amplitude*sin(2*pi*f*t);
v_aaf = aaf(v_mic,p);
v_pga = pga(v_aaf,p);
[phi,~] = vtc_bpf(v_pga,p);
pwm = xor_pwm(phi,p);
[events,~,~] = iaf(pwm,p);
keep = t >= 0.01 & t < duration;
basis = [sin(2*pi*f*t(keep)),cos(2*pi*f*t(keep)),ones(nnz(keep),1)];
fit = basis\phi(keep);
m.phase_amplitude = hypot(fit(1),fit(2));
m.duty = mean(pwm(keep));
m.duty_reference = mean(abs(phi(keep)))/pi;
samples_per_period = round(p.fs/p.f_inj);
assert(samples_per_period == p.fs/p.f_inj,'Carrier period must align to grid.');
pv = pwm(keep); ph = phi(keep);
n_periods = floor(numel(pv)/samples_per_period);
pv = reshape(pv(1:n_periods*samples_per_period),samples_per_period,n_periods);
ph = reshape(ph(1:n_periods*samples_per_period),samples_per_period,n_periods);
m.window_duty_mae = mean(abs(mean(pv,1)-mean(abs(ph),1)/pi));
gap = p.vhi-p.vlo; slope = p.I_iaf/p.C_iaf;
active_distance = sum(double(pwm(1:end-1)))/p.fs*slope;
m.event_count = numel(events);
m.analytic_event_count = floor((active_distance+gap)/(2*gap));
s = struct('t',t(keep),'phi',phi(keep),'pwm',pwm(keep));
end
