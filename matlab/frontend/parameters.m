function p = parameters()
% PARAMETERS  所有参数集中在这里；除 Q 外均为演示假设，不是论文器件提取值。
p.fs = 10e6;         % Hz，数值时间网格；不是芯片上的音频 ADC
p.f_aaf = 4000;      % Hz，单通道 AAF 截止（假设）
p.q_aaf = 1/sqrt(2); % 二阶低通阻尼（假设）
p.gain_db = 0;       % 固定 PGA 增益；论文可编程范围 0..12 dB，暂不做 AGC
p.vcm = 0.2;         % V，共模仅作工作点说明；模型电压均表示围绕共模的小信号
p.f_res = 1000;      % Hz，演示通道音频中心频率
p.Q = 4.5;          % 论文实测平均 Q，作为参考目标
p.G = 20;           % rad/V，ILRO 的小信号相位增益（假设，不随输入归一化）
p.N = 4;            % 反馈支路并联倍数（假设）
p.wp = 2*pi*p.f_res/p.Q;                   % rad/s，ILRO 一阶相位响应极点
p.Gint = (2*pi*p.f_res)^2/(p.wp*p.N*p.G);   % V/(rad*s)，CP 积分增益
p.f_inj = 50e3;      % Hz，本地注入载波（假设），与 f_res 严格分开
p.I_iaf = 2e-9;     % A，PWM 有效时的净充/放电电流幅值（假设）
p.C_iaf = 1e-12;    % F，IAF 电容（假设）
p.vlo = 0.08;       % V，理想迟滞比较器低阈值（假设）
p.vhi = 0.13;       % V，理想迟滞比较器高阈值（假设）
end
