function [phi, v_aaf, v_pga, centers] = frontend_bank16(v_mic_group, p)
% FRONTEND_BANK16  Four independent AAF/PGA instances feeding 16 VTC channels.
% v_mic_group is N-by-4: in normal use all columns carry the same microphone
% voltage, but separate columns make group-state independence testable.
assert(size(v_mic_group,2) == 4, 'Expected one microphone input per AAF/PGA group.');
centers = logspace(log10(125),log10(5000),16);
v_aaf = zeros(size(v_mic_group)); v_pga = v_aaf;
for group = 1:4
    v_aaf(:,group) = aaf(v_mic_group(:,group),p);
    v_pga(:,group) = pga(v_aaf(:,group),p);
end
phi = zeros(size(v_mic_group,1),16);
for channel = 1:16
    group = ceil(channel/4);
    pc = p; pc.f_res = centers(channel);
    pc.wp = 2*pi*pc.f_res/pc.Q;
    pc.Gint = (2*pi*pc.f_res)^2/(pc.wp*pc.N*pc.G);
    phi(:,channel) = vtc_bpf(v_pga(:,group),pc);
end
end
