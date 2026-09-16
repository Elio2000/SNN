function result = verify_single_tone_observability(run_dir, figure_dir)
% VERIFY_SINGLE_TONE_OBSERVABILITY  Audit exposed nodes while only tone frequency varies.
assert(~isfile(fullfile(run_dir,'observability.mat')), ...
    'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end
p = parameters(); amplitude = 0.03; duration = 0.05; start = 0.02;
frequencies = 1000*2.^[-1,-0.5,-0.25,0,0.25,0.5,1];
spp = p.fs/p.f_inj;
assert(spp == round(spp),'Carrier periods must align to the numerical grid.');

% Independent constant-phase contract for the ideal edge-pair PD.
pd_contract_error = zeros(1,3); phases = [-pi/4,0,pi/4];
for j = 1:3
    [~,vilro,vref] = xor_pwm(phases(j)*ones(round(0.01*p.fs),1),p);
    [up,dn] = pd_updn(vilro,vref);
    pd_contract_error(j) = abs((mean(up)-mean(dn))-phases(j)/(2*pi));
end

n = numel(frequencies);
phase_amplitude = zeros(1,n); phase_reconstruction_corr = zeros(1,n);
phase_sign_agreement = zeros(1,n); frequency_estimate = zeros(1,n);
carrier_rate = zeros(1,n); pd_signed_mean = zeros(1,n);
xor_duty = zeros(1,n); iaf_rate = zeros(1,n);
example = struct();
for j = 1:n
    f = frequencies(j); t = (0:round(duration*p.fs))'/p.fs;
    vmic = amplitude*sin(2*pi*f*t);
    phi = vtc_bpf(pga(aaf(vmic,p),p),p);
    [pwm,vilro,vref] = xor_pwm(phi,p);
    [up,dn] = pd_updn(vilro,vref);
    events = iaf(pwm,p);
    keep = t >= start & t < duration;
    basis = [sin(2*pi*f*t(keep)),cos(2*pi*f*t(keep)),ones(nnz(keep),1)];
    fit = basis\phi(keep); phase_amplitude(j) = hypot(fit(1),fit(2));
    ph = reshape(phi(keep),spp,[]);
    pu = reshape(up(keep),spp,[]); pd = reshape(dn(keep),spp,[]);
    phase_cycle = mean(ph,1);
    signed_cycle = (sum(pu,1)-sum(pd,1))/spp;
    reconstructed = 2*pi*signed_cycle;
    c = corrcoef(phase_cycle,reconstructed);
    phase_reconstruction_corr(j) = c(1,2);
    valid = abs(phase_cycle) > 2*pi/spp;
    phase_sign_agreement(j) = mean(sign(phase_cycle(valid)) == sign(signed_cycle(valid)));
    code = sign(signed_cycle); code(code == 0) = NaN;
    code = fillmissing(code,'previous'); code = fillmissing(code,'next');
    transitions = find(diff(code) ~= 0)+1;
    frequency_estimate(j) = p.f_inj/(2*median(diff(transitions)));
    segment = vilro(keep);
    carrier_rate(j) = sum(diff(segment) > 0)/(duration-start);
    pd_signed_mean(j) = mean(signed_cycle);
    xor_duty(j) = mean(pwm(keep));
    iaf_rate(j) = sum(events >= start & events < duration)/(duration-start);
    if f == 1000
        cycle_time = start+(0:numel(signed_cycle)-1)'/p.f_inj;
        example = struct('cycle_time',cycle_time,'phase_cycle',phase_cycle(:), ...
            'reconstructed',reconstructed(:),'signed_cycle',signed_cycle(:));
    end
end
frequency_relative_error = abs(frequency_estimate./frequencies-1);
carrier_relative_error = abs(carrier_rate/p.f_inj-1);
pd_contract_pass = max(pd_contract_error) <= 2*p.f_inj/p.fs+1e-9;
phase_readout_pass = min(phase_reconstruction_corr) > 0.95 && ...
    min(phase_sign_agreement) > 0.95;
frequency_pass = max(frequency_relative_error) < 0.02;
carrier_pass = max(carrier_relative_error) < 0.002;
zero_mean_pass = max(abs(pd_signed_mean)) < 0.002;
result = struct('parameters',p,'amplitude_vpeak',amplitude, ...
    'frequencies_hz',frequencies,'pd_contract_error',pd_contract_error, ...
    'phase_amplitude_rad',phase_amplitude, ...
    'phase_reconstruction_corr',phase_reconstruction_corr, ...
    'phase_sign_agreement',phase_sign_agreement, ...
    'frequency_estimate_hz',frequency_estimate, ...
    'frequency_relative_error',frequency_relative_error, ...
    'carrier_rate_hz',carrier_rate,'carrier_relative_error',carrier_relative_error, ...
    'pd_signed_mean',pd_signed_mean,'xor_duty',xor_duty,'iaf_rate_hz',iaf_rate, ...
    'pd_contract_pass',pd_contract_pass,'phase_readout_pass',phase_readout_pass, ...
    'frequency_pass',frequency_pass,'carrier_pass',carrier_pass, ...
    'zero_mean_pass',zero_mean_pass);
result.pass = pd_contract_pass && phase_readout_pass && frequency_pass && ...
    carrier_pass && zero_mean_pass;
fprintf('MATLAB %s; %s\n',version,computer);
fprintf('Frequencies [Hz]:'); fprintf(' %.9g',frequencies); fprintf('\n');
fprintf('PD contract errors:'); fprintf(' %.9g',pd_contract_error); fprintf('\n');
fprintf('Phase amplitudes [rad]:'); fprintf(' %.9g',phase_amplitude); fprintf('\n');
fprintf('PD phase correlations:'); fprintf(' %.9g',phase_reconstruction_corr); fprintf('\n');
fprintf('PD sign agreements:'); fprintf(' %.9g',phase_sign_agreement); fprintf('\n');
fprintf('Frequency estimates [Hz]:'); fprintf(' %.9g',frequency_estimate); fprintf('\n');
fprintf('Frequency max relative error: %.12g\n',max(frequency_relative_error));
fprintf('Carrier-rate max relative error: %.12g\n',max(carrier_relative_error));
fprintf('PD signed-mean max magnitude: %.12g\n',max(abs(pd_signed_mean)));
fprintf('XOR duties:'); fprintf(' %.9g',xor_duty); fprintf('\n');
fprintf('IAF rates [event/s]:'); fprintf(' %.9g',iaf_rate); fprintf('\n');
fprintf('PD/phase/frequency/carrier/zero-mean checks: %d %d %d %d %d\n', ...
    pd_contract_pass,phase_readout_pass,frequency_pass,carrier_pass,zero_mean_pass);
fprintf('Single-tone observability checks: %d\n',result.pass);
fprintf('PD polarity is instantaneous signal phase, not BPF tuning direction.\n');
save(fullfile(run_dir,'observability.mat'),'result','example');
summary = table(frequencies(:),phase_amplitude(:),phase_reconstruction_corr(:), ...
    phase_sign_agreement(:),frequency_estimate(:),frequency_relative_error(:), ...
    carrier_rate(:),pd_signed_mean(:),xor_duty(:),iaf_rate(:), ...
    'VariableNames',{'tone_hz','phase_amplitude_rad','pd_phase_corr', ...
    'pd_sign_agreement','pd_frequency_hz','pd_frequency_relative_error', ...
    'vilro_carrier_rate_hz','pd_signed_mean','xor_duty','iaf_event_rate_hz'});
writetable(summary,fullfile(run_dir,'summary.csv'));
fig = figure('Visible','off','Position',[100 100 1150 820]);
tiledlayout(3,1,'Padding','compact','TileSpacing','compact');
nexttile; semilogx(frequencies,phase_amplitude,'o-','LineWidth',1.3); hold on;
semilogx(frequencies,xor_duty,'x--','LineWidth',1.2);
xlabel('Tone frequency [Hz]'); ylabel('Magnitude statistic'); grid on;
legend('signed phase amplitude [rad]','XOR mean duty','Location','best');
title('E007 magnitude nodes show band response, not a unique tuning direction');
nexttile; semilogx(frequencies,frequency_estimate,'o-','LineWidth',1.3); hold on;
semilogx(frequencies,frequencies,'k--'); grid on;
xlabel('True tone frequency [Hz]'); ylabel('PD transition estimate [Hz]');
legend('Up/Dn estimate','ideal','Location','northwest');
nexttile; keep_example = example.cycle_time < start+0.004;
plot((example.cycle_time(keep_example)-start)*1e3, ...
    example.phase_cycle(keep_example),'LineWidth',1.3); hold on;
stairs((example.cycle_time(keep_example)-start)*1e3, ...
    example.reconstructed(keep_example),'LineWidth',1.0); grid on;
xlabel('Time after 20 ms [ms]'); ylabel('Phase [rad]');
legend('internal phase truth','2pi(UP-DN duty)','Location','best');
title('PD reconstruction uses only exposed carrier edges and 50-kHz timing');
exportgraphics(fig,fullfile(figure_dir,'single_tone_observability.png'),'Resolution',160);
close(fig);
assert(result.pass,'Single-tone node observability audit failed.');
end
