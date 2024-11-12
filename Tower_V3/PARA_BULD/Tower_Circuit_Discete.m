function [ TowerData] = Tower_Circuit_Discete( Tower, GLB )
% Generate left side of the MNA equation for time-series iteration 
%  Modify the MNA equation by including 
%   (1) self R arising from recurrsive convolution of VF results
%   (2) add current-measurement branch for connecting wire to Line Model
%       meas bran = Line node -> Tower node
%       Cw.id =[S/C, Span_id, CK_id, in/out, tower_nid], 
%       Cw.C0;
%   (3) TML eqn in PEEC model (hybrid)
%       -- sequential order: = order of common node in Tower.Node.com 
%                            = Span_id (tail/head) -> Cir_id -> Phase_id
dT = GLB.dT;
Info = Tower.Info;  
Bran = Tower.Bran;
Node = Tower.Node;
Para = Tower.CK_Para;

A1 = Para.A;           % full matrix of incidence
R = Para.R;            % Lumped/PEEC elements of Resistnace                   
L = Para.L;            % Lumped?PEEC elements of Inductance                   
P = Para.P;            % P matrix without cosnidering Pa and Pg
G = Para.G;            % G matrix without cosnidering Gg
C = Para.C;            % Lumped elements of Capacitance   
Cw = Para.Cw;          % ID & Cap of OHL for connection to tower   
Cwc = Para.Cwc;        % ID & Cap of CAB for connection to tower   
Ht = Para.Ht;          % residuals and poles of VF rational model function

Vmod = [Info{1,8:9}];   % [CondImpVF GndPEEC LossyGndCha]; 1=yes, 0=no
Nn = Node.num(1);                  % # of total nodes
Nb = Bran.num(1);                  % # of total brans

% (1) Update R by inclduing recurrsive convolution terms
Kt.ord = 0;
Kt.id  = Ht.id;
if ~isempty(Ht.id)
    if Vmod(1)==1
        Kt.b = exp(-Ht.r*GLB.dT);   % recurrsive conolution B = exp(-di*dt)
        Kt.a = (1-Kt.b).*Ht.r./Ht.d;    % A = ri/di*(1-exp(-di*dt))                
        R(Ht.id)=R(Ht.id)+sum(Kt.a,2);  % inclduing sum(Ak)term (2)
        
        Kt.id=Ht.id;                % range of PEEC wires (air + gnd): VF
        Kt.ord = size((Ht.r),2);
    end
end

% (2) Getting B matrix Y=G+i*B by including C 
B1 = inv(P) + C;                    % B matrix (Y=G+i*B)
B1 = B1/dt;

% (3) measurement branches for CTs and OHL/CAB connection 
% (3a) Span
Nr2= size(Cw.id,1);                  % # of all lines
A2 = zeros(Nr2,Nn);
B2 = zeros(Nr2,Nn);
C0 = GLB.slg*Cw.C0;                 % dx*C0

% nid = Cw.id(:,4);                   % tower node id: injected node
% A2(1:Nr,nid) = 1;
% B2(1:Nr,nid) = C0; 
for ik=1:Nr2     
    nid = Cw.id(ik,4);            % tower node id: injected node
    A2(ik,nid) = 1;  
    B2(ik,nid) = C0(ik);  
end
B2 = B2/dT;
Isef  = 2*Cw.id(:,4);               % coef. of past-time OHL current


% (3b) Cable
Nr3= size(Cwc.id,1);                % # of all lines
A3 = zeros(Nr3,Nn);
B3 = zeros(Nr3,Nn);
C0c = GLB.slg*Cwc.C0;               % dx*C0
Isefc = [];

if Nr3~=0
    nid = Cwc.id(:,4);              % tower node id: injected node
    A3(1:Nr3,nid) = 1;
    B3(1:Nr3,nid) = C0c; 
    B3 = B3/dT;
    Isefc = 2*Cwe.id(:,4);          % coef. of past-time CAB current
end
A2 = [A2; A3];
B2 = [B2; B3];

% (4) Generate inductive/capacitve parameters of the equation
X  = L/dT;
A0 = zeros(Nb,Nr2+Nr3);  
E0 = eye(Nr2+Nr3);

% (5) Generate parameters of the equation (left side) for Veq
LEFT=[-A1   -(R+X)  A0;             % create the matrix (left)
       G+B1 -A1'   -A2';
       B2    A0'    E0];
LEFT=inv(LEFT);

%**************************************************************************
%(6) Output Tower Data
Cal.At = [A1 A2]';                  % transpose incident matrix[+/-b1 +/-b2
Cal.LEFT = LEFT;
Cal.X= X;
Cal.B1 = B1;
Cal.B2 = B2;
Cal.Kt = Kt;
Cal.ord = Kt.ord;
Cal.Isef = Isef;
Cal.Isefc = Isefc;
Tower.Cal = Cal;
end