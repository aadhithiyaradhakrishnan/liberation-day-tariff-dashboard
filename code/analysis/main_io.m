
clc
clearvars

data = readtable('../../data/base_data/trade_cepii.csv');
X_ji=table2array(data);
X_ji(isnan(X_ji))=0;
N = size(X_ji,1);
id_US = 185;

% GDP data
gdp = readtable('../../data/base_data/gdp.csv');
Y_i=table2array(gdp);
Y_i=Y_i/1000; %trade flows are in 1000 of USD

tot_exports=sum(X_ji,2);
tot_imports=sum(X_ji,1)';

nu_eq = solveNu(X_ji,Y_i,id_US);
nu=nu_eq(1)*ones(N,1); nu(id_US) = nu_eq(2);

T = (1-nu).*(sum(X_ji,1)' - sum(repmat((1-nu)',N,1).*X_ji,2));
E_i = Y_i + T;
X_ii = E_i - tot_imports; 
X_ii(X_ii<0)=0; 
X_ji(eye(N)==1) = X_ii;


beta = 0.49; nu_IO = nu;
X_ji_IO = X_ji;
X_ji_IO(eye(N)==1) = X_ji(eye(N)==1)/beta;
E_i_IO = sum(X_ji_IO,1)';
Y_i_IO = beta*sum( repmat((1-nu_IO)',N,1).*X_ji_IO,2) + nu_IO.*sum(X_ji_IO,1)';
lambda_ji_IO = X_ji_IO./repmat(E_i_IO',N,1);
T_IO = E_i_IO - (Y_i_IO + (1-beta)*sum( repmat((1-nu_IO)',N,1).*X_ji_IO,2)); 
 
lambda_ji = X_ji./repmat(E_i',N,1);


%--- Read USTR tariffs
reuters = readtable('../../data/base_data/tariffs.csv');
new_ustariff=table2array(reuters);
t_ji = zeros(N,N);
t_ji(:,id_US)=new_ustariff;

t_ji(:,id_US)=max(0.1, t_ji(:,id_US));
t_ji(id_US,id_US) = 0 ;
tariff{1} = t_ji;

eps = 4;
kappa = 0.5;
psi = 0.67/eps;

theta = eps/0.67;
phi_tilde = (1+theta)./((1-nu)*theta) - (1/theta) - 1;
Phi = {1 + phi_tilde, 0.5+ phi_tilde};

results = zeros(N,7,4);
d_trade_IO = zeros(2);
d_employment_IO = zeros(2);

%--------- Roundabout Production --------
t_ji_new = tariff{1};
phi_IO = Phi{1}; 

data = {N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO};
param = {eps, kappa, psi, phi_IO, beta};

x0=[ ones(N,1); ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_IO(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve_1=fsolve(syst, x0, options);

[~,results(:,:,1), d_trade_IO(1)] = Balanced_Trade_IO(x_fsolve_1, data, param);
d_employment_IO(1) = sum(results(:,5,1).*Y_i_IO)./sum(Y_i_IO);
%-----------  Optimal tariff +IO --------

%----  find optimal tariff -----
tariff_case = 1;
t_ji_new =zeros(N,N);
data = {N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO};
param = {eps, kappa, psi, phi_IO, beta};
LB=[0.75*x_fsolve_1; 0*ones(N-1,1)];
UB=[1.5*x_fsolve_1; 0.25*ones(N-1,1)];
x0=[x_fsolve_1; 0.15*ones(N-1,1)];
target = @(x) obj_mpec(x, data, param, id_US, tariff_case);
constraint = @(x) const_mpec(x, data, param, id_US, tariff_case);
options = optimoptions(@fmincon,'Display','iter','MaxFunEvals',inf,...
        'MaxIter',5000,'TolFun',1e-8,'TolX',1e-8, 'TolCon', 1e-8, 'algorithm','interior-point' );
[x_fmincon, ~]=fmincon(target,x0,[],[],[],[],LB,UB,constraint,options);
t_optimal = x_fmincon(end);
%---------------------------------

    t_ji_new =zeros(N,N);
    t_ji_new(:,id_US) =t_optimal;
    t_ji_new(id_US, id_US)=0;
data = {N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO};
param = {eps, kappa, psi, phi_IO, beta};

x0=[ ones(N,1); ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_IO(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve=fsolve(syst, x0, options);


[~,results(:,:,2), ~] = Balanced_Trade_IO(x_fsolve, data, param);

%-----------  Liberation Tariffs with optimal retaliation +IO --------
%----  find optimal tariff -----
tariff_case = 2;
t_ji_new =  tariff{1};
data = {N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO};
param = {eps, kappa, psi, phi_IO, beta};
LB=[0.75*x_fsolve_1; 0*ones(N-1,1)];
UB=[1.5*x_fsolve_1; 0.5*ones(N-1,1)];
x0=[x_fsolve_1; t_ji_new(setdiff(1:N,id_US),id_US)];
target = @(x) obj_mpec(x, data, param, id_US, tariff_case);
constraint = @(x) const_mpec(x, data, param, id_US, tariff_case);
options = optimoptions(@fmincon,'Display','iter','MaxFunEvals',inf,...
        'MaxIter',200,'TolFun',1e-6,'TolX',1e-6, 'TolCon', 1e-6, 'algorithm','interior-point' );
[x_fmincon, ~]=fmincon(target,x0,[],[],[],[],LB,UB,constraint,options);
t_optimal = x_fmincon(4*N+1:end);
%---------------------------------

t_ji_new =  tariff{1};
t_ji_new(id_US,setdiff(1:N,id_US)) = t_optimal;
t_ji_new(id_US,id_US) = 0;

data = {N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO};
param = {eps, kappa, psi, phi_IO, beta};

x0=[ ones(N,1); ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_IO(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,4), ~] = Balanced_Trade_IO(x_fsolve, data, param);

%-----------  Liberation Tariffs with reciprocal retaliation + IO --------
t_ji_new =  tariff{1}; 
t_ji_new(id_US,:) = t_ji_new(:,id_US)';
t_ji_new(id_US,id_US) = 0;

data = {N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO};
param = {eps, kappa, psi, phi_IO, beta};

x0=[ ones(N,1); ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_IO(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10); %, 'Algorithm','levenberg-marquardt'
x_fsolve=fsolve(syst, x0, options);

[~, results(:,:,3), d_trade_IO(2) ] = Balanced_Trade_IO(x_fsolve, data, param);
d_employment_IO(2) = sum(results(:,5,3).*Y_i_IO)./sum(Y_i_IO);


%-----------  Multi-Sector IO Model --------
run sub_multisector_io.m
N_multi = N; N = 194;
load("../../output/Table_11.mat", "d_trade", "d_employment");

%---- print table of results ------
run print_tables_io.m


%% Functions

function [ceq, results, d_trade] = Balanced_Trade_IO(x, data, param)

[N, E_i, Y_i, lambda_ji, t_ji, nu, T_i] = data{:};
[eps, kappa, psi, phi, beta] = param{:};

w_i_h=abs(x(1:N));    % abs(.) is used avoid complex numbers...
E_i_h=abs(x(N+1:N+N));
L_i_h = abs(x(N+N+1:N+N+N));
P_i_h = abs(x(N+N+N+1:N+N+N+N));
% construct 2D matrix from 1D vector
%wi_h_2D = repmat(w_i_h,1,N);
phi_2D = repmat(phi',N,1);

% construct new trade values
c_i_h=repmat((w_i_h.^beta).*(P_i_h.^(1-beta)), [1 N]);
%entry = (w_i_h/P_i_h).^(1-beta);
%p_ij_h = ( (c_i_h./((entry*L_i_h).^psi) ).^-eps ) .* ((1+t_ji).^(-eps*phi_2D));
entry = repmat((w_i_h./P_i_h).^(1-beta), [1 N]);
p_ij_h = ( (c_i_h./((entry.*L_i_h).^psi) ).^-eps ) .* ((1+t_ji).^(-eps*phi_2D));
AUX0 = lambda_ji.* p_ij_h;
AUX1 = repmat(sum(AUX0,1), N,1);
lambda_ji_new = AUX0./AUX1;
Y_i_h= w_i_h.*L_i_h;
Y_i_new= Y_i_h.*Y_i;
E_i_new = E_i .* E_i_h;


X_ji_new = lambda_ji_new.* repmat(E_i_new', N, 1)./(1+t_ji);
tariff_rev = sum(lambda_ji_new.*(t_ji./(1+t_ji)).*repmat(E_i_new', N, 1))';

tau_i = tariff_rev./Y_i_new;
tau_i_new = 0 ;
tau_i_h = (1 -  tau_i_new) ./(1-tau_i);
%tau_i = 0; tau_i_h =1;
% ------------------------------------------------------------------
%        Wage Income = Total Sales net of Taxes
% ------------------------------------------------------------------
nu_2D = repmat(nu',N,1);
ERR1 = sum(beta*(1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;
ERR1(N,1) = mean((w_i_h-1).*Y_i);  % replace one excess equation 

% ------------------------------------------------------------------
%        Total Income = Total Sales 
% ------------------------------------------------------------------
X_global = sum(Y_i);
X_global_new = sum(Y_i_new);

ERR2 = tariff_rev + (w_i_h.*L_i_h.*Y_i) + sum((1-beta)*(1-nu_2D).*X_ji_new,2) + T_i.*(X_global_new./X_global)  -  E_i_new;

% ------------------------------------------------------------------

ERR3 = L_i_h - (tau_i_h.*w_i_h./P_i_h).^kappa;

ERR4 = P_i_h - ( (E_i_h./w_i_h).^(1 - phi)) .* ( sum(AUX0,1).^(-1./eps)');
ceq= [ERR1' ERR2' ERR3' ERR4'];

Ec_i = Y_i + T_i;
delta_i = Ec_i./(Ec_i - kappa*(1-tau_i).*Y_i/(1+kappa));
Ec_i_h = (tariff_rev + (w_i_h.*L_i_h.*Y_i) + T_i.*(X_global_new./X_global))./Ec_i;
W_i_h = delta_i .* (Ec_i_h ./ P_i_h) + (1-delta_i).*(w_i_h.*L_i_h ./ P_i_h);
%W_i_h = (E_i_h ./ P_i_h);

% factual trade flows
X_ji = lambda_ji.*repmat(E_i',N,1);
D_i =  sum(X_ji,1)' - sum(X_ji,2) ;
D_i_new =  sum(X_ji_new,1)' - sum(X_ji_new,2);


d_welfare = 100*(W_i_h-1);
d_export = 100*( (sum(X_ji_new.*(1-eye(N)),2)./Y_i_new)./ (sum(X_ji.*(1-eye(N)),2)./Y_i) - 1);
d_import = 100*( (sum(X_ji_new.*(1-eye(N)),1)./Y_i_new')./ (sum(X_ji.*(1-eye(N)),1)./Y_i') - 1)';
d_employment = 100*(L_i_h - 1);
d_CPI = 100*(P_i_h - 1);
d_D_i = 100*((D_i_new - D_i)./abs(D_i));

results = [d_welfare d_D_i  d_export d_import d_employment d_CPI tariff_rev./E_i];

trade = X_ji.*(1-eye(N));
trade_new = X_ji_new.*(1+t_ji).*(1-eye(N));
d_trade = 100*( (sum(trade_new(:))./ sum(trade(:)))./ (sum(Y_i_new(:))/sum(Y_i(:)) ) - 1);

end



function [c, ceq] = const_mpec(x, data, param, id, tariff_case)

[N, E_i, Y_i, lambda_ji, t_ji, nu, T_i] = data{:};
[eps, kappa, psi, phi, beta] = param{:};

w_i_h=abs(x(1:N));    % abs(.) is used avoid complex numbers...
E_i_h=abs(x(N+1:N+N));
L_i_h = abs(x(N+N+1:N+N+N));
P_i_h = abs(x(N+N+N+1:N+N+N+N));
t = abs(x(N+N+N+N+1:end));

if tariff_case ==1
t_ji(setdiff(1:N,id),id) = t;
elseif tariff_case ==2
t_ji(id,setdiff(1:N,id))= t;
end
t_ji(eye(N)==1)=0;
% construct 2D matrix from 1D vector
phi_2D = repmat(phi',N,1);

% construct new trade values
c_i_h=repmat((w_i_h.^beta).*(P_i_h.^(1-beta)), [1 N]);
entry = repmat((w_i_h./P_i_h).^(1-beta), [1 N]);
p_ij_h = ( (c_i_h./((entry.*L_i_h).^psi) ).^-eps ) .* ((1+t_ji).^(-eps*phi_2D));
AUX0 = lambda_ji.* p_ij_h;
AUX1 = repmat(sum(AUX0,1), N,1);
lambda_ji_new = AUX0./AUX1;
Y_i_h= w_i_h.*L_i_h;
Y_i_new= Y_i_h.*Y_i;
E_i_new = E_i .* E_i_h;


X_ji_new = lambda_ji_new.* repmat(E_i_new', N, 1)./(1+t_ji);
tariff_rev = sum(lambda_ji_new.*(t_ji./(1+t_ji)).*repmat(E_i_new', N, 1))';

tau_i = tariff_rev./Y_i_new;
tau_i_new = 0 ;
tau_i_h = (1 -  tau_i_new) ./(1-tau_i);
% ------------------------------------------------------------------
%        Wage Income = Total Sales net of Taxes
% ------------------------------------------------------------------
nu_2D = repmat(nu',N,1);
ERR1 = sum(beta*(1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;
ERR1(N,1) = mean((w_i_h-1).*Y_i);  % replace one excess equation 

% ------------------------------------------------------------------
%        Total Income = Total Sales 
% ------------------------------------------------------------------
X_global = sum(Y_i);
X_global_new = sum(Y_i_new);

ERR2 = tariff_rev + (w_i_h.*L_i_h.*Y_i) + sum((1-beta)*(1-nu_2D).*X_ji_new,2) + T_i.*(X_global_new./X_global)  -  E_i_new;

% ------------------------------------------------------------------

ERR3 = L_i_h - (tau_i_h.*w_i_h./P_i_h).^kappa;

ERR4 = P_i_h - ( (E_i_h./w_i_h).^(1 - phi)) .* ( sum(AUX0,1).^(-1./eps)');
ceq= [ERR1' ERR2' ERR3' ERR4'];
c = [];

end


function [gains] = obj_mpec(x, data, param, id, tariff_case)

[N, E_i, Y_i, lambda_ji, t_ji, nu, T_i] = data{:};
[eps, kappa, psi, phi, beta] = param{:};

w_i_h=abs(x(1:N));    % abs(.) is used avoid complex numbers...
E_i_h=abs(x(N+1:N+N));
L_i_h = abs(x(N+N+1:N+N+N));
P_i_h = abs(x(N+N+N+1:N+N+N+N));
t = abs(x(N+N+N+N+1:end));

if tariff_case ==1
t_ji(setdiff(1:N,id),id) = t;
elseif tariff_case ==2
t_ji(id,setdiff(1:N,id))= t;
end
t_ji(eye(N)==1)=0;

phi_2D = repmat(phi',N,1);

% construct new trade values
c_i_h=repmat((w_i_h.^beta).*(P_i_h.^(1-beta)), [1 N]);
entry = repmat((w_i_h./P_i_h).^(1-beta), [1 N]);
p_ij_h = ( (c_i_h./((entry.*L_i_h).^psi) ).^-eps ) .* ((1+t_ji).^(-eps*phi_2D));
AUX0 = lambda_ji.* p_ij_h;
AUX1 = repmat(sum(AUX0,1), N,1);
lambda_ji_new = AUX0./AUX1;
Y_i_h= w_i_h.*L_i_h;
Y_i_new= Y_i_h.*Y_i;
E_i_new = E_i .* E_i_h;
tariff_rev = sum(lambda_ji_new.*(t_ji./(1+t_ji)).*repmat(E_i_new', N, 1))';

tau_i = tariff_rev./Y_i_new;

X_global = sum(Y_i);
X_global_new = sum(Y_i_new);

Ec_i = Y_i + T_i;
delta_i = Ec_i./(Ec_i - kappa*(1-tau_i).*Y_i/(1+kappa));
Ec_i_h = (tariff_rev + (w_i_h.*L_i_h.*Y_i) + T_i.*(X_global_new./X_global))./Ec_i;
W_i_h = delta_i .* (Ec_i_h ./ P_i_h) + (1-delta_i).*(w_i_h.*L_i_h ./ P_i_h);

    if tariff_case ==1
        gains = -100*(W_i_h(id)-1);
    elseif tariff_case == 2
        gains = -100* sum( Y_i([1:id-1, id+1:end]).*(W_i_h([1:id-1 id+1:end])-1)) ...
                                                            ./sum(Y_i([1:id-1, id+1:end]));  
    end   

end



function nu = solveNu(X,Y, id_US)

N = size(X, 1);  % assuming X is an n x n matrix

AggI = [ones(1,N); zeros(1,N)]; 
AggI(1,id_US) = 0; AggI(2,id_US) = 1;
X = AggI*X*AggI';
Y = AggI*Y;

% Initial guess for nu: uniform distribution
nu0 = [0.1; 0.24];

% Set options for fsolve (you can change Display to 'off' if desired)
options = optimoptions('fsolve', 'Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');

% Use fsolve to solve the system of equations defined in eqFun
nu = fsolve(@(nu) eqFun(nu, X, Y), nu0, options);
nu(nu<0) = 0;

end

function F = eqFun(nu, X, Y)
% eqFun defines the system of equations for nu.

E_i = Y + (1-nu).*(sum(X,1)' - sum(repmat((1-nu)',2,1).*X,2)) ;

r_11 = (E_i(1) - X(2,1))/(E_i(1) - X(2,1) + X(1,2)); 
r_22 = (E_i(2) - X(1,2))/(E_i(2) - X(1,2)+ X(2,1));

F(1) = (1-r_11)*nu(2) +  r_11*nu(1) - 0.12;
F(2) = r_22*nu(2) +  (1-r_22)*nu(1) - 0.26;


end