function result = verify_pd_span_readout(run_dir, figure_dir)
% VERIFY_PD_SPAN_READOUT  Replace single-interval median by multi-edge span.
assert(~isfile(fullfile(run_dir,'span_readout.mat')), ...
    'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end
p = parameters(); amplitude = 0.03; duration = 0.05; start = 0.02;
frequencies = 1000*2.^[-1,-0.5,-0.25,0,0.25,0.5,1];
spp = p.fs/p.f_inj; n = numel(frequencies);
estimate = zeros(1,n); relative_error = zeros(1,n);
balanced_mean = zeros(1,n); span_s = zeros(1,n); intervals = zeros(1,n);
for j = 1:n
    f = frequencies(j); t = (0:round(duration*p.fs))'/p.fs;
    phi = vtc_bpf(pga(aaf(amplitude*sin(2*pi*f*t),p),p),p);
    [~,vilro,vref] = xor_pwm(phi,p); [up,dn] = pd_updn(vilro,vref);
    keep = t >= start & t < duration;
    pu = reshape(up(keep),spp,[]); pd = reshape(dn(keep),spp,[]);
    signed_cycle = (sum(pu,1)-sum(pd,1))/spp;
    code = sign(signed_cycle); code(code == 0) = NaN;
    code = fillmissing(code,'previous'); code = fillmissing(code,'next');
    transitions = find(diff(code) ~= 0)+1;
    intervals(j) = 2*floor((numel(transitions)-1)/2); % integer tone periods
    first = transitions(1); last = transitions(1+intervals(j));
    span_s(j) = (last-first)/p.f_inj;
    estimate(j) = intervals(j)/(2*span_s(j));
    relative_error(j) = abs(estimate(j)/f-1);
    balanced_mean(j) = mean(signed_cycle(first:last-1));
end
frequency_pass = max(relative_error) < 0.02;
window_pass = min(span_s) >= 0.025 && max(abs(balanced_mean)) < 0.002;
result = struct('parameters',p,'amplitude_vpeak',amplitude, ...
    'frequencies_hz',frequencies,'estimate_hz',estimate, ...
    'relative_error',relative_error,'balanced_pd_mean',balanced_mean, ...
    'span_s',span_s,'half_cycle_intervals',intervals, ...
    'frequency_pass',frequency_pass,'window_pass',window_pass, ...
    'pass',frequency_pass && window_pass);
fprintf('MATLAB %s; %s\n',version,computer);
fprintf('Frequencies [Hz]:'); fprintf(' %.9g',frequencies); fprintf('\n');
fprintf('Span estimates [Hz]:'); fprintf(' %.9g',estimate); fprintf('\n');
fprintf('Relative errors:'); fprintf(' %.9g',relative_error); fprintf('\n');
fprintf('Observation spans [s]:'); fprintf(' %.9g',span_s); fprintf('\n');
fprintf('Balanced PD means:'); fprintf(' %.9g',balanced_mean); fprintf('\n');
fprintf('Frequency/window checks: %d %d\n',frequency_pass,window_pass);
fprintf('PD multi-transition span checks: %d\n',result.pass);
fprintf('This readout adds transition count, elapsed-cycle state, and an integer-period window.\n');
save(fullfile(run_dir,'span_readout.mat'),'result');
summary = table(frequencies(:),estimate(:),relative_error(:),span_s(:), ...
    intervals(:),balanced_mean(:),'VariableNames',{'tone_hz','estimate_hz', ...
    'relative_error','observation_span_s','half_cycle_intervals','balanced_pd_mean'});
writetable(summary,fullfile(run_dir,'summary.csv'));
fig = figure('Visible','off','Position',[100 100 1050 700]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
nexttile; semilogx(frequencies,estimate,'o-','LineWidth',1.3); hold on;
semilogx(frequencies,frequencies,'k--'); grid on;
xlabel('True tone frequency [Hz]'); ylabel('Span estimate [Hz]');
legend('multi-transition span','ideal','Location','northwest');
title('E008 same PD ports and input; readout estimator only');
nexttile; semilogx(frequencies,100*relative_error,'o-','LineWidth',1.3); hold on;
yline(2,'r--','2% gate'); grid on;
xlabel('Tone frequency [Hz]'); ylabel('Relative error [%]');
title('Integer-period window also removes fixed-window signed-mean bias');
exportgraphics(fig,fullfile(figure_dir,'pd_span_readout.png'),'Resolution',160);
close(fig);
assert(result.pass,'PD span readout audit failed.');
end
