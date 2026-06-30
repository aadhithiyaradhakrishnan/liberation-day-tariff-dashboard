clc
clearvars
disp(['pwd = ', pwd]);
%----- read trade and GDP data ---------
data = readtable('../../data/base_data/trade_cepii.csv');
X_ji=table2array(data);
X_ji(isnan(X_ji))=0;
N = size(X_ji,1);
id_US = 185;
id_CAN = 31;
id_MEX = 115;
id_CHN = 34;
id_EU=[10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, 88, 107, 108, 109, 119, 133, 144, 145, 149, 164, 165];
id_RoW = setdiff(1:N, [id_US, id_CHN, id_EU]);
non_US = setdiff(1:N, id_US);

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


E_i = sum(X_ji,1)';
Y_i = sum( repmat((1-nu)',N,1).*X_ji,2) + nu.*sum(X_ji,1)';
T = E_i - Y_i;
lambda_ji = X_ji./repmat(E_i',N,1);


cases = {'benchmark','partial_passthrough'};

%--- Read US tariffs
reuters = readtable('../../data/base_data/tariffs.csv');
new_ustariff=table2array(reuters);
t_ji = zeros(N,N);
t_ji(:,id_US)=new_ustariff;

t_ji(:,id_US)=max(0.1, t_ji(:,id_US));
t_ji(id_US,id_US) = 0 ;
tariff{1} = t_ji;

%trade elasticity
eps = 4;
kappa = 0.5;
psi = 0.67/eps;

theta = eps/0.67;
phi_tilde = (1+theta)./((1-nu)*theta) - (1/theta) - 1;

Phi = {1 + phi_tilde, 0.5+ phi_tilde, 0.25+ phi_tilde};


%-- create array to save results
results = zeros(N,7,9);
revenue = zeros(1,4);
d_trade = zeros(1,9);
d_employment = zeros(1,9);

%% ------------------------------------------------------------------------------
 %                              Baseline Analysis
% ------------------------------------------------------------------------------
for i = 1:2

t_ji_new =  tariff{1}; % use Rueters

    if i == 1
        phi = Phi{i};
    elseif  i == 2  
        phi = Phi{i+1};
    end    

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};
lump_sum = 0;

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,i), d_trade(i)] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

revenue(i) = results(id_US,7,i);
d_employment(i) = sum(results(:,5,i).*Y_i)./sum(Y_i);
end


%---- Eaton-Kortum Specification ----
Y_i_EK = sum(X_ji,2);
T_EK = E_i - Y_i_EK;

t_ji_new =  tariff{1}; % use Rueters
phi = 1;
nu_EK =zeros(N,1);

data = {N, E_i, Y_i_EK, lambda_ji, t_ji_new, nu_EK, T_EK};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,3), d_trade(3)] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

revenue(3) = results(id_US,7,3);
d_employment(3) = sum(results(:,5,3).*Y_i)./sum(Y_i);

%---- Lump-sum rebate of tariff revenue ----
t_ji_new =  tariff{1}; % use USTR tariffs
phi = Phi{1};

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};
lump_sum = 1;

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,8), ~] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

%---- Lump-sum rebate of tariff revenue ----
t_ji_new =  tariff{1}; % use USTR tariffs
phi = Phi{1};

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {2*eps, kappa, psi, phi};
lump_sum = 0;

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,9), ~] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);


%----------- Optimal Tariff w/o retaliation --------

t_ji_new =  zeros(N,N);
phi = Phi{1};

delta = ( sum(X_ji.*repmat((1-nu)',N,1).*(1-eye(N)).*(1-lambda_ji),2)) ./((1-nu).*(E_i-X_ji(eye(N)==1)));
t_ji_new(:,id_US) = 1/((1 + delta(id_US)*eps)*phi(id_US) -  1); 
t_ji_new(id_US,id_US) = 0;

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};
lump_sum = 0;

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,4), d_trade(4)] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

revenue(4) = results(id_US,7,4);
d_employment(4) = sum(results(:,5,4).*Y_i)./sum(Y_i);

%-----------  Liberation Tariffs with optimal retaliation --------

t_ji_new =  tariff{1};

phi = Phi{1};

AggI = [ones(1,N); zeros(1,N)]; 
AggI(1,id_US) = 0; AggI(2,id_US) = 1;
X = AggI*X_ji*AggI';
Y = AggI*Y_i;
lambda = X./repmat(sum(X),2,1);
delta = ( sum(X.*repmat((1-nu_eq)',2,1).*(1-eye(2)).*(1-lambda),2)) ./((1-nu_eq).*(Y-X(eye(2)==1)));

t_ji_new(id_US,:) = 1./((1+delta(1)*eps)*phi(1)' - 1);
t_ji_new(id_US,id_US) = 0;

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,5), d_trade(5)] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

revenue(5) = results(id_US,7,5);
d_employment(5) = sum(results(:,5,5).*Y_i)./sum(Y_i);

%-----------  Liberation Tariffs with reciprocal retaliation --------
t_ji_new =  tariff{1};

phi = Phi{1};

t_ji_new(id_US,:) = t_ji_new(:,id_US)';
t_ji_new(id_US,id_US) = 0;

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,6), d_trade(6)] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

revenue(6) = results(id_US,7,6);
d_employment(6) = sum(results(:,5,6).*Y_i)./sum(Y_i);

%----------- Optimal Tariff w/ optimal retaliation --------

t_ji_new =  zeros(N,N);
phi = Phi{1};

delta = ( sum(X_ji.*repmat((1-nu)',N,1).*(1-eye(N)).*(1-lambda_ji),2)) ./((1-nu).*(Y_i-X_ji(eye(N)==1)));
t_ji_new(:,id_US) = 1/((1 + delta(id_US)*eps)*phi(id_US) -  1); 
t_ji_new(id_US,id_US) = 0;


AggI = [ones(1,N); zeros(1,N)]; 
AggI(1,id_US) = 0; AggI(2,id_US) = 1;
X = AggI*X_ji*AggI';
Y = AggI*Y_i;
lambda = X./repmat(sum(X),2,1);
delta = ( sum(X.*repmat((1-nu_eq)',2,1).*(1-eye(2)).*(1-lambda),2)) ./((1-nu_eq).*(Y-X(eye(2)==1)));

t_ji_new(id_US,:) = 1./((1+delta(1)*eps)*phi(1)' - 1);
t_ji_new(id_US,id_US) = 0;

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param, lump_sum);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,7), d_trade(7)] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum);

d_employment(7) = sum(results(:,5,7).*Y_i)./sum(Y_i);

%------------- save results -------
countries = readtable('../../data/base_data/country_labels.csv');
country_names = countries.iso3; % Adjust 'Country' to match actual column name in CSV

Data_base = results(:,:,1);
Tab = table(country_names, Data_base, 'VariableNames', {'Country', 'Value'});
writetable(Tab, '../../output/output_map.csv');

Data_retal = results(:,:,5);
Tab = table(country_names, Data_retal, 'VariableNames', {'Country', 'Value'});
writetable(Tab, '../../output/output_map_retal.csv');


%---- Multi-Sector Model ------
run sub_multisector_baseline 
N_multi = N; N = 194;
save("../../output/Table_11.mat", "d_trade", "d_employment"); 


%---- print table of results ------
run print_tables_baseline.m

%----------------------------------------        
function [ceq, results, d_trade] = Balanced_Trade_EQ(x, data, param, lump_sum)

[N, E_i, Y_i, lambda_ji, t_ji, nu, T_i] = data{:};
[eps, kappa, psi, phi] = param{:};

w_i_h=abs(x(1:N));    % abs(.) is used avoid complex numbers...
E_i_h=abs(x(N+1:N+N));
L_i_h = abs(x(N+N+1:N+N+N));
% construct 2D matrix from 1D vector
wi_h_2D = repmat(w_i_h,1,N);

phi_2D = repmat(phi',N,1);
% construct new trade values
AUX0 = lambda_ji.* ( (wi_h_2D./(L_i_h.^psi)).^-eps ) .* ((1+t_ji).^(-eps*phi_2D));
AUX1 = repmat(sum(AUX0,1), N,1);
lambda_ji_new = AUX0./AUX1;
Y_i_h= w_i_h.*L_i_h;
Y_i_new= Y_i_h.*Y_i;
E_i_new = E_i .* E_i_h;


P_i_h=( (E_i_h./w_i_h).^(1 - phi)) .* ( sum(AUX0,1).^(-1./eps)');

X_ji_new = lambda_ji_new.* repmat(E_i_new', N, 1)./(1+t_ji);
tariff_rev = sum(lambda_ji_new.*(t_ji./(1+t_ji)).*repmat(E_i_new', N, 1))';

if lump_sum == 0

    tau_i = tariff_rev./Y_i_new;
    tau_i_new = 0 ;
    tau_i_h = (1 -  tau_i_new) ./(1-tau_i);

elseif lump_sum == 1

    tau_i = 0; tau_i_h=1;

end
% ------------------------------------------------------------------
%        Wage Income = Total Sales net of Taxes
% ------------------------------------------------------------------
nu_2D = repmat(nu',N,1);
ERR1 = sum((1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;
ERR1(N,1) = mean((P_i_h-1).*E_i);  % replace one excess equation 

% ------------------------------------------------------------------
%        Total Income = Total Sales 
% ------------------------------------------------------------------
X_global = sum(Y_i);
X_global_new = sum(Y_i_new);

ERR2 = tariff_rev + (w_i_h.*L_i_h.*Y_i)  + T_i.*(X_global_new./X_global)  -  E_i_new;

% ------------------------------------------------------------------

ERR3 = L_i_h - (tau_i_h.*w_i_h./P_i_h).^kappa;

ceq= [ERR1' ERR2' ERR3'];

delta_i = E_i./(E_i - kappa*(1-tau_i).*Y_i/(1+kappa));
W_i_h = delta_i .* (E_i_h ./ P_i_h) + (1-delta_i).*(w_i_h.*L_i_h ./ P_i_h);

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

trade = X_ji.*(1-eye(N));
trade_new = X_ji_new.*(1+t_ji).*(1-eye(N));
d_trade = 100*( (sum(trade_new(:))./ sum(trade(:)))./ (sum(Y_i_new(:))/sum(Y_i(:)) ) - 1);

results = [d_welfare d_D_i  d_export d_import d_employment d_CPI tariff_rev./E_i];
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
r_22 = (E_i(2) - X(1,2))/(E_i(2) - X(1,2) + X(2,1));

F(1) = (1-r_11)*nu(2) +  r_11*nu(1) - 0.12;
F(2) = r_22*nu(2) +  (1-r_22)*nu(1) - 0.26;


end