clc
clearvars -except results revenue d_trade_IO d_employment_IO Y_i E_i ...
            Phi nu psi kappa id_US id_EU id_CHN id_RoW non_US country_names

data = readtable('../../data/ITPDS/trade_ITPD.csv');  % Reads as table
X = table2array(data(:,4));
N = 194;
K = 4;
X_ji = reshape(X, N,N,K);


t = readtable('../../data/base_data/tariffs.csv');
new_ustariff=table2array(t);
id_US = 185;
t_ji = zeros(N,N,K);
t_ji(:,id_US,1:K-1)=repmat(new_ustariff, [1 1 K-1]);

t_ji(:,id_US, 1:K-1)=max(0.1, t_ji(:,id_US, 1:K-1));
t_ji(id_US,id_US, 1:K-1) = 0 ;

problematic_id = sum(all(X_ji == 0, 1),3);
ID = find(problematic_id == 1);
idx = setdiff(1:N, ID);  
N = numel(idx);
X_new = zeros(N, N, K);
t_new = zeros(N, N, K);
for k = 1:K
    X_new(:,:,k) = X_ji(idx, idx, k);  % drop row and column ID
    t_new(:,:,k) = t_ji(idx, idx, k);

end
X_ji = X_new;
t_ji = t_new;

id_US_new  = find(idx == 185);


% id_CAN = find(idx == 31);
% id_MEX = find(idx == 115);
% id_CHN = find(idx == 34);
% id_EU = find(ismember(idx, [10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, 88, 107, 108, 109, 119, 133, 144, 145, 149, 164, 165]));
% id_RoW = setdiff(1:N, [id_US, id_CHN, id_EU]);
% non_US = setdiff(1:N, id_US);

%nu_eq = [0.1143 0.2705];
%nu=nu_eq(1)*ones(N,1); nu(id_US_new) = nu_eq(2);
%kappa = 0.5;
%psi = 0.67/4;
%theta = 1/psi;
%phi = (1+theta)./((1-nu)*theta) - (1/theta) - 1/2 ;

phi = Phi{2}; phi=phi(idx); nu = nu(idx);
beta = [0.51 0.32 0.49 0.56];
beta_3D = repmat(permute(beta,[1 3 2]), N, N);

nu_3D = repmat(nu',N,1,K);
E_i_multi = sum(sum(X_ji,1),3)';
Y_i_multi = sum(sum((1-nu_3D).*beta_3D.*X_ji,3) , 2) + sum(sum(nu_3D.*X_ji,1),3)';
T = E_i_multi - (Y_i_multi + sum(sum((1-beta_3D).*(1-nu_3D).*X_ji,3),2));
lambda_ji = X_ji./repmat(sum(X_ji),[N 1 1]);
e_i = repmat(sum(X_ji,1),[N 1 1])./repmat(E_i_multi',[N 1 K]);

Y_ik_p = sum( (1-nu_3D).* beta_3D.*X_ji , 2) ;
Y_ik_f = permute(sum(nu_3D.*X_ji,1), [2 1 3]);
Y_ik = Y_ik_p + Y_ik_f;
ell_ik = Y_ik./repmat( Y_i_multi, [1 1 K]);


phi_avg=sum(Phi{1}.*Y_i)./sum(Y_i);
eps = [3.3 3.8 4.1]/phi_avg; eps(K)=3;
eps_3D = repmat(permute(eps', [3 2 1]), [N N 1]);

results_multi = zeros(N,7,2);
d_trade_IO_multi = zeros(2);
d_employment_IO_multi = zeros(2);

%--------- No Retaliation -------------
data = {N, K, E_i_multi, Y_i_multi, lambda_ji, e_i, ell_ik, t_ji, nu, T};
param = {eps_3D, kappa, psi, phi, beta_3D};

x0=[ ones(N,1); ones(N,1); ones(N,1); ones(N,1); ones(N*K,1)];

syst=@(x) Balanced_Trade_IO(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',15,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results_multi(:,:,1), d_trade_IO_multi(1)] = Balanced_Trade_IO(x_fsolve, data, param);
d_employment_IO_multi(1) = sum(results_multi(:,5,1).*Y_i_multi)./sum(Y_i_multi);
%----------- Reciprocal Retaliation --------
for k = 1:K-1
t_ji(id_US_new,:,k) = t_ji(:,id_US_new,k)';
end
t_ji(id_US_new,id_US_new,:) = 0;

data = {N, K, E_i_multi, Y_i_multi, lambda_ji, e_i, ell_ik, t_ji, nu, T};
param = {eps_3D, kappa, psi, phi, beta_3D};

syst=@(x) Balanced_Trade_IO(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',50,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve=fsolve(syst, x_fsolve, options);

[~,results_multi(:,:,2), d_trade_IO_multi(2)] = Balanced_Trade_IO(x_fsolve, data, param);
d_employment_IO_multi(2) = sum(results_multi(:,5,2).*Y_i_multi)./sum(Y_i_multi);
%}
%----------------------------------------        

function [ceq, results, d_trade] = Balanced_Trade_IO(x, data, param)

[N, K, E_i, Y_i, lambda_ji, e_i, ell_ik, t_ji, nu, T_i] = data{:};
[eps, kappa, psi, phi, beta] = param{:};

w_i_h=abs(x(1:N));    % abs(.) is used avoid complex numbers...
E_i_h=abs(x(N+1:N+N));
L_i_h = abs(x(N+N+1:N+N+N));
P_i_h = abs(x(N+N+N+1:N+N+N+N));
ell_ik_h = reshape(abs(x(N+N+N+N+1:end)), [N 1 K]);

% construct 3D matrix from 1D vector
wi_h_3D = repmat(w_i_h,[1 N K]);
Pi_h_3D = repmat(P_i_h,[1 N K]);
Lik_h_3D = repmat(L_i_h, [1 N K]).* repmat( ell_ik_h, [1 N 1]);
phi_3D = repmat(phi',N,1,K);

% construct new trade values
c_i_h=(wi_h_3D.^beta).*(Pi_h_3D.^(1-beta));
entry = repmat((w_i_h./P_i_h), [1 N K]).^(1-beta);

p_ij_h = ( (c_i_h./((entry.*Lik_h_3D).^psi) ).^-eps ) .* ((1+t_ji).^(-eps.*phi_3D));
AUX0 = lambda_ji.* p_ij_h;
AUX1 = repmat(sum(AUX0,1), N,1,1);
lambda_ji_new = AUX0./AUX1;

Y_i_h= w_i_h.*L_i_h;
Y_i_new= Y_i_h.*Y_i;
E_i_new = E_i .* E_i_h;


X_ji_new = lambda_ji_new.* e_i.* repmat(E_i_new', N, 1, K)./(1+t_ji);

tariff_rev = sum(sum(t_ji.*X_ji_new),3)';

tau_i = tariff_rev./Y_i;
tau_i_new = 0 ;
tau_i_h = (1 -  tau_i_new) ./(1-tau_i);
%tau_i = 0; tau_i_h =1;
% ------------------------------------------------------------------
%        Wage Income = Total Sales net of Taxes
% ------------------------------------------------------------------
nu_3D = repmat(nu',N,1,K);
Y_ik_h = wi_h_3D(:,1,:).*Lik_h_3D(:,1,:);
Y_ik = ell_ik.*repmat(Y_i, [1 1 K]);
Y_ik_cf = sum((1-nu_3D).*beta.*X_ji_new,2) + permute(sum(nu_3D.*X_ji_new,1), [2 1 3]);
ERR1 =  reshape(Y_ik_cf - Y_ik.* Y_ik_h, N*K,1);
ERR1(N,1) = sum((E_i_h-1).*E_i); % replace one excess equation
% ------------------------------------------------------------------
%        Total Income = Total Sales 
% ------------------------------------------------------------------
X_global = sum(Y_i);
X_global_new = sum(Y_i_new);

ERR2 = tariff_rev + (w_i_h.*L_i_h.*Y_i) + sum(sum((1-beta).*(1-nu_3D).*X_ji_new,2),3) + T_i.*(X_global_new./X_global)  -  E_i_new;

% ------------------------------------------------------------------

ERR3 = L_i_h - (tau_i_h.*w_i_h./P_i_h).^kappa;

ERR4 = P_i_h - ( (E_i_h./w_i_h).^(1-phi)) .* prod( sum(AUX0,1).^(-e_i(1,:,:)./eps(1,:,:)) ,3)';

ERR5 = reshape(100*(sum(ell_ik.*ell_ik_h,3)-1), N,1);

ceq= [ERR1' ERR2' ERR3' ERR4' ERR5'];

Ec_i = Y_i + T_i;
delta_i = Ec_i./(Ec_i - kappa*(1-tau_i).*Y_i/(1+kappa));
Ec_i_h = (tariff_rev + (w_i_h.*L_i_h.*Y_i) + T_i.*(X_global_new./X_global))./Ec_i;
W_i_h = delta_i .* (Ec_i_h ./ P_i_h) + (1-delta_i).*(w_i_h.*L_i_h ./ P_i_h);

% factual trade flows
X_ji = lambda_ji.*e_i.*repmat(E_i',N,1);
D_i =  sum(sum(X_ji,1),3)' - sum(sum(X_ji,2),3) ;
D_i_new =  sum(sum(X_ji_new,1),3)' - sum(sum(X_ji_new,2),3);

d_welfare = 100*(W_i_h-1);
d_export = 100*( (sum(sum(X_ji_new,3).*(1-eye(N)),2)./Y_i_new)./ (sum(sum(X_ji,3).*(1-eye(N)),2)./Y_i) - 1);
d_import = 100*( (sum(sum(X_ji_new,3).*(1-eye(N)),1)./Y_i_new')./ (sum(sum(X_ji,3).*(1-eye(N)),1)./Y_i') - 1)';
d_employment = 100*(L_i_h - 1);
d_CPI = 100*(P_i_h - 1);
d_D_i = 100*((D_i_new - D_i)./abs(D_i));
 
results = [d_welfare d_D_i  d_export d_import d_employment d_CPI tariff_rev./E_i];

trade = X_ji.*(1-eye(N));
trade_new = X_ji_new.*(1+t_ji).*repmat((1-eye(N)),[1 1 K]);
d_trade = 100*( (sum(trade_new(:))./ sum(trade(:)))./ (sum(Y_i_new(:))/sum(Y_i(:)) ) - 1);

end

