function result = verify_filterbank16(run_dir, figure_dir)
% VERIFY_FILTERBANK16  Intrinsic response and four-group state audit.
assert(~isfile(fullfile(run_dir,'filterbank16.mat')), ...
    'Use a new run directory; preserve previous results.');
if ~isfolder(run_dir), mkdir(run_dir); end
if ~isfolder(figure_dir), mkdir(figure_dir); end
p = parameters();
centers = logspace(log10(125),log10(5000),16);
groups = ceil((1:16)/4);
spacing = diff(log(centers));
spacing_error = max(abs(spacing-mean(spacing)));
endpoint_relative_error = max(abs([centers(1)/125-1,centers(end)/5000-1]));

frequency = logspace(log10(80),log10(8000),20001)';
intrinsic = zeros(numel(frequency),16);
full_chain = intrinsic;
peak_hz = zeros(1,16); measured_q = zeros(1,16);
full_gain_at_center = zeros(1,16);
w = 2*pi*frequency;
wa = 2*pi*p.f_aaf;
h_aaf = wa^2./((1i*w).^2+(wa/p.q_aaf)*(1i*w)+wa^2);
pga_gain = 10^(p.gain_db/20);
for channel = 1:16
    w0 = 2*pi*centers(channel); wp = w0/p.Q;
    gint = w0^2/(wp*p.N*p.G);
    h = p.G*(1i*w)./(p.N*gint*p.G+1i*w+(1i*w).^2/wp);
    intrinsic(:,channel) = abs(h);
    full_chain(:,channel) = abs(h.*h_aaf)*pga_gain;
    [peak,index] = max(intrinsic(:,channel)); peak_hz(channel) = frequency(index);
    level = peak/sqrt(2);
    below = find(intrinsic(1:index,channel) < level,1,'last');
    above_offset = find(intrinsic(index:end,channel) < level,1,'first');
    above = index+above_offset-1;
    flo = interp1(intrinsic([below,below+1],channel),frequency([below,below+1]),level);
    fhi = interp1(intrinsic([above-1,above],channel),frequency([above-1,above]),level);
    measured_q(channel) = peak_hz(channel)/(fhi-flo);
    full_gain_at_center(channel) = interp1(frequency,full_chain(:,channel),centers(channel));
end
peak_relative_error = abs(peak_hz./centers-1);
q_relative_error = abs(measured_q/p.Q-1);

crossover_db = zeros(15,2);
for channel = 1:15
    fmid = sqrt(centers(channel)*centers(channel+1));
    for side = 0:1
        c = channel+side; w0 = 2*pi*centers(c); wp = w0/p.Q;
        gint = w0^2/(wp*p.N*p.G); s = 1i*2*pi*fmid;
        h = p.G*s/(p.N*gint*p.G+s+s^2/wp);
        crossover_db(channel,side+1) = 20*log10(abs(h)/p.G);
    end
end

% Four AAF/PGA copies and all 16 VTC states must remain independent.
ps = p; ps.fs = 200e3;
t = (0:round(0.01*ps.fs))'/ps.fs;
base = 0.001*sin(2*pi*500*t)+0.0005*sin(2*pi*1200*t);
inputs = repmat(base,1,4);
[phase_a,aaf_a,pga_a] = frontend_bank16(inputs,ps);
cut = floor(numel(t)/2); inputs_b = inputs;
inputs_b(cut+1:end,2) = 0;
[phase_b,aaf_b,pga_b] = frontend_bank16(inputs_b,ps);
unaffected = [1:4,9:16]; affected = 5:8;
other_groups_exact = isequal(phase_a(:,unaffected),phase_b(:,unaffected)) && ...
    isequal(aaf_a(:,[1,3,4]),aaf_b(:,[1,3,4])) && ...
    isequal(pga_a(:,[1,3,4]),pga_b(:,[1,3,4]));
affected_prefix_exact = isequal(phase_a(1:cut,affected),phase_b(1:cut,affected));
affected_suffix_changes = max(abs(phase_a(cut+1:end,affected)- ...
    phase_b(cut+1:end,affected)),[],'all') > 0;
identical_group_frontends = isequal(aaf_a(:,1),aaf_a(:,2),aaf_a(:,3),aaf_a(:,4)) && ...
    isequal(pga_a(:,1),pga_a(:,2),pga_a(:,3),pga_a(:,4));

distribution_pass = endpoint_relative_error < 1e-12 && spacing_error < 1e-12;
response_pass = max(peak_relative_error) < 0.002 && max(q_relative_error) < 0.02;
crossover_pass = all(crossover_db > -4 & crossover_db < -3,'all');
independence_pass = other_groups_exact && affected_prefix_exact && ...
    affected_suffix_changes && identical_group_frontends;
result = struct('parameters',p,'centers_hz',centers,'groups',groups, ...
    'peak_hz',peak_hz,'measured_q',measured_q, ...
    'peak_relative_error',peak_relative_error,'q_relative_error',q_relative_error, ...
    'crossover_db',crossover_db,'full_gain_at_center',full_gain_at_center, ...
    'endpoint_relative_error',endpoint_relative_error, ...
    'log_spacing_error',spacing_error,'other_groups_exact',other_groups_exact, ...
    'affected_prefix_exact',affected_prefix_exact, ...
    'affected_suffix_changes',affected_suffix_changes, ...
    'identical_group_frontends',identical_group_frontends, ...
    'distribution_pass',distribution_pass,'response_pass',response_pass, ...
    'crossover_pass',crossover_pass,'independence_pass',independence_pass);
result.pass = distribution_pass && response_pass && crossover_pass && independence_pass;

fprintf('MATLAB %s; %s\n',version,computer);
fprintf('Centers [Hz]:'); fprintf(' %.9g',centers); fprintf('\n');
fprintf('Peak max relative error: %.12g\n',max(peak_relative_error));
fprintf('Measured Q range: %.12g to %.12g; max relative error %.12g\n', ...
    min(measured_q),max(measured_q),max(q_relative_error));
fprintf('Adjacent crossover dB range: %.12g to %.12g\n', ...
    min(crossover_db,[],'all'),max(crossover_db,[],'all'));
fprintf('Full-chain center gain range [rad/V]: %.12g to %.12g\n', ...
    min(full_gain_at_center),max(full_gain_at_center));
fprintf('State checks other/prefix/change/identical frontends: %d %d %d %d\n', ...
    other_groups_exact,affected_prefix_exact,affected_suffix_changes,identical_group_frontends);
fprintf('Distribution/response/crossover/independence checks: %d %d %d %d\n', ...
    distribution_pass,response_pass,crossover_pass,independence_pass);
fprintf('16-channel reference checks: %d\n',result.pass);
fprintf('All four AAF copies use the same assumed 4-kHz cutoff; full gain is not normalized.\n');

save(fullfile(run_dir,'filterbank16.mat'),'result','frequency','intrinsic', ...
    'full_chain','phase_a','phase_b','aaf_a','aaf_b','pga_a','pga_b');
summary = table((1:16)',groups(:),centers(:),peak_hz(:),measured_q(:), ...
    peak_relative_error(:),q_relative_error(:),full_gain_at_center(:), ...
    'VariableNames',{'channel','aaf_pga_group','center_hz','peak_hz', ...
    'measured_q','peak_relative_error','q_relative_error', ...
    'full_chain_center_gain_rad_per_v'});
writetable(summary,fullfile(run_dir,'summary.csv'));

fig = figure('Visible','off','Position',[100 100 1150 820]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
nexttile;
semilogx(frequency,20*log10(intrinsic/p.G),'LineWidth',1.0); grid on;
xlim([100,6500]); ylim([-22,1]);
xlabel('Frequency [Hz]'); ylabel('Intrinsic VTC gain [dB, G normalized]');
title('E006 16 log-spaced channels, Q = 4.5');
nexttile;
semilogx(frequency,20*log10(full_chain/p.G),'LineWidth',1.0); grid on;
xlim([100,6500]); ylim([-25,1]);
xlabel('Frequency [Hz]'); ylabel('Full-chain gain [dB relative to G]');
title('Unnormalized response with four independent 4-kHz AAF/PGA copies');
exportgraphics(fig,fullfile(figure_dir,'filterbank16.png'),'Resolution',160);
close(fig);
assert(result.pass,'16-channel reference audit failed.');
end
