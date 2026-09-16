function verify_frontend()
% VERIFY_FRONTEND  独立解析参考 + 因果性 + 输入尺度 + 边沿/电荷守恒。
% Gate 在 E001 登记；这是工程数值检查，不是论文硅测复现。
p = parameters(); p.fs = 1e6; % 相位测试网格，独立于演示载波网格
frequencies = [250, 1000, 4000];
t = (0:round(0.08*p.fs))'/p.fs;
for f = frequencies
    v = 0.01*sin(2*pi*f*t);
    phi = vtc_bpf(v,p);
    keep = t >= 0.04;
    basis = [sin(2*pi*f*t(keep)), cos(2*pi*f*t(keep)), ones(nnz(keep),1)];
    fit = basis\phi(keep);
    measured = (fit(1)+1i*fit(2))/0.01;
    s = 1i*2*pi*f;
    expected = p.G*s/(p.N*p.Gint*p.G+s+s^2/p.wp);
    err = abs(measured-expected)/abs(expected);
    fprintf('Phase gain at %g Hz: complex relative error %.6g\n',f,err);
    assert(err < 0.02, 'Phase transfer mismatch.');
end
v = 0.01*sin(2*pi*1000*t);
phi = vtc_bpf(v,p); phi2 = vtc_bpf(2*v,p);
assert(max(abs(phi2-2*phi)) < 1e-12, 'Gain depends on waveform amplitude.');
v2 = v; cut = floor(numel(v)/2); v2(cut+1:end) = 0;
phi2 = vtc_bpf(v2,p);
assert(isequal(phi(1:cut),phi2(1:cut)), 'Noncausal phase output.');
a1 = aaf(v,p); a2 = aaf(v2,p);
assert(isequal(a1(1:cut),a2(1:cut)), 'Noncausal AAF output.');
a_dc = aaf(0.01*ones(size(t)),p);
assert(abs(a_dc(end)-0.01) < 1e-9, 'AAF DC gain mismatch.');
p.gain_db = 6;
assert(max(abs(pga(v,p)-10^(6/20)*v)) < 1e-12, 'PGA gain mismatch.');

% 电压翻倍应相位翻倍；超过线性相位范围必须报错而非偷偷截幅。
rejected = false;
try
    vtc_bpf(20*v,p);
catch err
    rejected = contains(err.message,'Phase exceeds');
end
assert(rejected, 'Out-of-range phase must be reported.');

% 时间步减半：比较同一时刻的相位；避免只验证离散公式自洽。
p2 = p; p2.fs = 2*p.fs;
t2 = (0:2*(numel(t)-1))'/p2.fs;
fine = vtc_bpf(0.01*sin(2*pi*1000*t2),p2);
err = max(abs(phi-fine(1:2:end)))/max(abs(fine));
fprintf('Phase time-step refinement relative error %.6g\n',err);
assert(err < 0.005, 'Phase time grid has not converged.');

p = parameters();
for phase = [-pi/4, 0, pi/4]
    n = round(0.01*p.fs);
    pwm = xor_pwm(phase*ones(n,1),p);
    duty = mean(pwm);
    fprintf('XOR phase %.5g rad: duty %.6g (expected %.6g)\n',phase,duty,abs(phase)/pi);
    assert(abs(duty-abs(phase)/pi) <= 2*p.f_inj/p.fs+1e-9, 'XOR duty mismatch.');
    if phase == 0, assert(~any(pwm), 'Zero phase creates false PWM.'); end
end

% IAF 理想迟滞振荡器的解析解：累计有效充电时间 -> 三角电压。
pwm = repmat([true(73,1); false(127,1)],500,1);
[events,voltage,spk] = iaf(pwm,p);
gap = p.vhi-p.vlo; slope = p.I_iaf/p.C_iaf;
distance = [0;cumsum(double(pwm(1:end-1)))]/p.fs*slope;
expected_v = p.vlo+gap-abs(mod(distance,2*gap)-gap);
expected_n = floor((distance(end)+gap)/(2*gap));
fprintf('IAF rising events %d (analytic %d), max voltage error %.6g V\n', ...
    numel(events),expected_n,max(abs(voltage-expected_v)));
assert(max(abs(voltage-expected_v)) < 1e-9, 'IAF charge conservation mismatch.');
assert(numel(events) == expected_n, 'IAF event count mismatch.');
assert(all(diff(events)>0), 'IAF events not chronological.');
assert(all(diff(voltage(1:100)) <= slope/p.fs+1e-12), 'IAF voltage step invalid.');
[quiet,voltage,spk] = iaf(false(1000,1),p);
assert(isempty(quiet) && all(voltage==p.vlo) && ~any(spk), 'IAF silence mismatch.');
fprintf('PASS: phase-domain transfer, amplitude, causality, refinement, XOR and IAF.\n');
end
