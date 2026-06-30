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


E_i = sum(X_ji,1)';
Y_i = sum( repmat((1-nu)',N,1).*X_ji,2) + nu.*sum(X_ji,1)';
T = E_i - Y_i;

lambda_ji = X_ji./repmat(E_i',N,1);


%--- Read USTR tariffs ------
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
phi = (1+theta)./((1-nu)*theta) - (1/theta) ;


%-- create array to save results
results = zeros(N,7,4);

%--------- Fixed Deficit ---------------
Y_i_EK = sum(X_ji,2);
T_EK = E_i - Y_i_EK;
nu_EK =0;

t_ji_new =  tariff{1}; % use Rueters
phi_EK = 1;

data = {N, E_i, Y_i_EK, lambda_ji, t_ji_new, nu_EK, T_EK};
param = {eps, kappa, psi, phi_EK};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,1)] = Balanced_Trade_EQ(x_fsolve, data, param);

%--------- Zero Deficit ---------------

%--- balance trade with the US
T_i_new = T - (X_ji(id_US,:)'-X_ji(:,id_US)); % subtract imbalance w/ US from deficit
T_i_new(id_US) = 0;

data = {N, E_i, Y_i, lambda_ji, zeros(N,N), nu, T_i_new};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve=fsolve(syst, x0, options);

[~,temp_a] = Balanced_Trade_EQ(x_fsolve, data, param);

t_ji_new =  tariff{1}; % use Rueters
data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T_i_new};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve=fsolve(syst, x0, options);

[~,temp_b] = Balanced_Trade_EQ(x_fsolve, data, param);

results(:,:,2) = temp_b - temp_a; 

%--------- Fixed Deficit + Retaliation---------------
Y_i_EK = sum(X_ji,2);
T_EK = E_i - Y_i_EK;


t_ji_new =  tariff{1};
t_ji_new(id_US,:) = 1/((1+eps)*phi_EK - 1);
t_ji_new(id_US,id_US) = 0;

data = {N, E_i, Y_i_EK, lambda_ji, t_ji_new, nu_EK, T_EK};
param = {eps, kappa, psi, phi_EK};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,3)] = Balanced_Trade_EQ(x_fsolve, data, param);


%--------- Zero Deficit + Retaliation ---------------
T_i_new = T - (X_ji(id_US,:)'-X_ji(:,id_US)); % subtract imbalance w/ US from deficit
T_i_new(id_US) = 0;


data = {N, E_i, Y_i, lambda_ji, zeros(N,N), nu, T_i_new};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve=fsolve(syst, x0, options);

[~,temp_a] = Balanced_Trade_EQ(x_fsolve, data, param);


t_ji_new =  tariff{1};
t_ji_new(id_US,:) = 1/((1+eps)*phi(id_US) - 1);
t_ji_new(id_US,id_US) = 0;

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T_i_new};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10);
x_fsolve=fsolve(syst, x0, options);

[~,temp_b] = Balanced_Trade_EQ(x_fsolve, data, param);

results(:,:,4) = temp_b - temp_a; 

% --------------------------------------------------------------------------------------
    %                       Print Output (Table 10)
  % --------------------------------------------------------------------------------------

countries = readtable('../../data/base_data/country_labels.csv');
country_names = countries.iso3; % Adjust 'Country' to match actual column name in CSV

tablePreamble = {...
'\begin{tabular}{lccccccc}';
        '\toprule';
        '\multicolumn{6}{l}{\textbf{(1) Pre-retaliation: fixed transfers to global GDP (Dekle et al., 2008) }}  \\';
        '\midrule';
        'Country &';
        '\specialcell{$\Delta$ welfare} &';
        '\specialcell{$\Delta$ $\frac{\textrm{exports}}{\textrm{GDP}}$} & ';
        '\specialcell{$\Delta$ $\frac{\textrm{imports}}{\textrm{GDP}}$} &';
        '\specialcell{$\Delta$ employment} &';
        '\specialcell{$\Delta$ prices} \\';

        '\midrule'
        };

   tableClosing = {...
        ' \bottomrule';
        '\end{tabular}'
        };

   fileID = fopen('../../output/Table_10.tex', 'w');

%%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end

%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 1));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 1));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 1));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 1));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 1));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,1)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,1));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));
        

        mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{(2) Pre-retaliation: balanced trade (Ossa, 2014) }} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 2));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 2));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 2));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 2));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 2));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,2)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,2));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));

         mid_table = {...            
        '\midrule';
        '\addlinespace[25pt]';
        '\multicolumn{6}{l}{\textbf{(3) Post-retaliation: fixed transfers to global GDP (Dekle et al., 2008) }} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 3));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 3));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 3));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 3));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 3));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,3)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,3));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));

mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{(4) Post-retaliation: balanced trade (Ossa, 2014) }} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 4));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 4));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 4));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 4));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 4));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,4)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,3));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));       


%%% TABLE CLOSING  %%%
for n = 1:numel(tableClosing)
    fprintf(fileID, '%s\n', tableClosing{n});
end

fclose(fileID);

%----------------------------------------        
function [ceq, results] = Balanced_Trade_EQ(x, data, param)

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


tau_i = tariff_rev./Y_i_new;
tau_i_new = 0 ;
tau_i_h = (1 -  tau_i_new) ./(1-tau_i);
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