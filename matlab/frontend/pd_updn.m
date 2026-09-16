function [up, dn] = pd_updn(v_ilro, v_ref)
% PD_UPDN  Edge-pair phase detector using only two exposed logic carriers.
% Convention: UP means Vilro rising edge arrived first; DN means Vref first.
% The earlier edge starts one pulse and the later edge resets it. Outputs are
% mutually exclusive. This ideal model omits reset delay, dead zone and glitches.
v_ilro = logical(v_ilro(:)); v_ref = logical(v_ref(:));
rise_ilro = [false; diff(v_ilro) > 0];
rise_ref = [false; diff(v_ref) > 0];
up = false(size(v_ilro)); dn = up; state = 0;
for k = 1:numel(v_ilro)
    if rise_ilro(k) && rise_ref(k)
        state = 0;
    elseif state == 0
        if rise_ilro(k), state = 1;
        elseif rise_ref(k), state = -1;
        end
    elseif state == 1 && rise_ref(k)
        state = 0;
    elseif state == -1 && rise_ilro(k)
        state = 0;
    end
    up(k) = state == 1; dn(k) = state == -1;
end
assert(~any(up & dn),'UP and DN must be mutually exclusive.');
end
