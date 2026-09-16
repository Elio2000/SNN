function [event_time, v_iaf, v_spk] = iaf(pwm, p)
% IAF  PWM 逻辑 -> 电容电压 [V]、迟滞比较器输出、上升沿事件时间 [s]。
% Fig.17.8.3 的理想充放电行为：EN=PWM，关断时保留电压和比较器状态。
% 低输出时净 +I/C 充电；到 vhi 翻高；高输出时 -I/C 放电；到 vlo 翻低。
% 不用 abs(phi) 直接造脉冲；不仿真比较器延迟、泄漏和晶体管开关细节。
dt = 1/p.fs;
slope = p.I_iaf/p.C_iaf;
assert(slope*dt < p.vhi-p.vlo, 'Reduce dt: at most one threshold crossing per step.');
v = p.vlo; high = false;
v_iaf = zeros(size(pwm)); v_spk = false(size(pwm));
event_time = zeros(numel(pwm),1); n = 0;
for k = 1:numel(pwm)
    v_iaf(k) = v; v_spk(k) = high;
    if k == numel(pwm) || ~pwm(k), continue; end
    before = v;
    if ~high
        v = v + slope*dt;
        if v >= p.vhi
            n = n+1;
            event_time(n) = (k-1)*dt + (p.vhi-before)/slope;
            v = 2*p.vhi-v; high = true; % 余下时间已经转为放电
        end
    else
        v = v - slope*dt;
        if v <= p.vlo, v = 2*p.vlo-v; high = false; end
    end
end
event_time = event_time(1:n);
end
