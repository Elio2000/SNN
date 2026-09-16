function [phi_next, in_locked_branch] = sc_ilro_cycle_map(phi, f0, f_inj, alpha)
% SC_ILRO_CYCLE_MAP  One normalized injection-cycle update on 0 <= phi <= pi.
% phi is the absolute SC-ILRO phase lag [rad] from the patent relation.
% The correction kernel is the inverse of EP4350992B1 [0103]-[0104]:
%   f0/f_inj = (1+alpha)/(alpha+phi/pi), alpha = CL/CI.
% It reproduces the published locked branch but is not a transistor model.
assert(alpha > 0 && f0 > 0 && f_inj > 0, 'Frequencies and alpha must be positive.');
natural_drift = 2*pi*(f0/f_inj-1);
injection_correction = 2*pi*((1+alpha)./(alpha+phi/pi)-1);
phi_next = phi-natural_drift+injection_correction;
in_locked_branch = phi >= 0 & phi <= pi & phi_next >= 0 & phi_next <= pi;
end
