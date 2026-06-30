clc
clearvars

data = readtable('../../data/base_data/trade_cepii.csv');
X_ji=table2array(data);
X_ji(isnan(X_ji))=0;
N = size(X_ji,1);
id_US = 185;
id_CHN = 34;
id_CAN = 31;
id_MEX = 115;
id_EU=[10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, ...
            88, 107, 108, 109, 119, 133, 144, 145, 149, 164, 165];
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

%--- Read USTR tariffs -----
reuters = readtable('../../data/base_data/tariffs.csv');
new_ustariff=table2array(reuters);
t_ji = zeros(N,N);
t_ji(:,id_US)=new_ustariff;

t_ji(:,id_US)=max(0.1, t_ji(:,id_US));
t_ji(id_US,id_US) = 0 ;
tariff_USTR = t_ji;

%trade elasticity
eps = 4;
kappa = 0.5;
psi = 0.67/eps;

theta = eps/0.67;
phi_tilde = (1+theta)./((1-nu)*theta) - (1/theta) - 1;

Phi = {1 + phi_tilde, 0.5+ phi_tilde};

%-- create array to save results
results = zeros(N,7,3);

%-----------  USTR tariff on China/EU + 10 percent tariff on others --------

t_ji_new =zeros(N,N);
t_ji_new(non_US,id_US) = 0.1; t_ji_new(id_US, non_US) = 0.1;

t_ji_new(id_CHN, id_US) = tariff_USTR(id_CHN, id_US);
t_ji_new(id_EU, id_US) = tariff_USTR(id_EU, id_US);

t_ji_new(id_US, id_CHN) = tariff_USTR(id_CHN, id_US)';
t_ji_new(id_US, id_EU) = tariff_USTR(id_EU, id_US)';
t_ji_new(id_US,id_US) =0;


phi = Phi{1};

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,1)] = Balanced_Trade_EQ(x_fsolve, data, param);



%-----------  USTR tariff on China + 10 percent tariff on others --------
t_ji_new =zeros(N,N);
t_ji_new(non_US,id_US) = 0.1; t_ji_new(id_US, non_US) = 0.1;

t_ji_new(id_CHN, id_US) = tariff_USTR(id_CHN, id_US);
t_ji_new(id_US, id_CHN) = tariff_USTR(id_CHN, id_US)';
t_ji_new(id_US,id_US) =0;

phi = Phi{1};


data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,2)] = Balanced_Trade_EQ(x_fsolve, data, param);



%------ 108 percent tariff on China + 10 percent tariff on others  --------

t_ji_new =zeros(N,N);
t_ji_new(non_US,id_US) = 0.1; t_ji_new(id_US, non_US) = 0.1;

t_ji_new(id_CHN, id_US) = 1.08;
t_ji_new(id_US, id_CHN) = 1.08;
t_ji_new(id_US,id_US) =0;
phi = Phi{1};

data = {N, E_i, Y_i, lambda_ji, t_ji_new, nu, T};
param = {eps, kappa, psi, phi};

x0=[ ones(N,1); ones(N,1); ones(N,1)];
syst=@(x) Balanced_Trade_EQ(x, data, param);
options = optimset('Display','iter','MaxFunEvals',inf,'MaxIter',inf,...
                'TolFun',1e-10,'TolX',1e-10, 'Algorithm','levenberg-marquardt');
x_fsolve=fsolve(syst, x0, options);

[~,results(:,:,3)] = Balanced_Trade_EQ(x_fsolve, data, param);


%------------- save results -------
countries = readtable('../../data/base_data/country_labels.csv');
country_names = countries.iso3; % Adjust 'Country' to match actual column name in CSV

% --------------------------------------------------------------------------------------
   %                 Print Output (Table 8)
% --------------------------------------------------------------------------------------

tablePreamble = {...
'\begin{tabular}{lcccccc}';
        '\toprule';
        '\multicolumn{4}{l}{\textbf{Case 1: US trade war with EU \& China}}  \\';
        '\midrule';
        'Country &';
        '\specialcell{$\Delta$ welfare} &';
        '\specialcell{$\Delta$ deficit} &';
        '\specialcell{$\Delta$ employment} &';
        '\specialcell{$\Delta$ prices} \\';

        '\midrule'
        };

   tableClosing = {...
        ' \bottomrule';
        '\end{tabular}'
        };

   fileID = fopen('../../output/Table_8.tex', 'w');

%%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end

%%%  COLUMNS WITH RESULTS %%%
for i = [id_US, id_CHN]
    fprintf(fileID, '%s & ', country_names{i});
    fprintf(fileID, '%1.2f\\%% & ', results(i, 1, 1));
    fprintf(fileID, '%1.1f\\%% & ', results(i, 2, 1));
    fprintf(fileID, '%1.2f\\%% & ', results(i, 5, 1));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(i, 6, 1));
    fprintf(fileID, ' \\addlinespace[3pt]\n');
end
%%%  WRITE AVERAGES for EU %%%
        avg_EU = sum(E_i(id_EU,:).*results(id_EU,:,1)) ...
                                                         ./sum(E_i(id_EU,:));
        %avg_non_US(1) = mean(results(id_EU,1,1));
        
        fprintf(fileID, 'EU & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_EU(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_EU(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_EU(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_EU(6));
        fprintf(fileID, ' \\addlinespace[3pt]\n');

       
        avg_RoW = sum(E_i(id_RoW,:).*results(id_RoW,:,1))./sum(E_i(id_RoW,:));
        
        fprintf(fileID, 'RoW & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_RoW(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_RoW(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_RoW(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_RoW(6)); 


        mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{4}{l}{\textbf{Case 2: US trade war with China}}  \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
for i = [id_US, id_CHN]
    fprintf(fileID, '%s & ', country_names{i});
    fprintf(fileID, '%1.2f\\%% & ', results(i, 1, 2));
    fprintf(fileID, '%1.1f\\%% & ', results(i, 2, 2));
    fprintf(fileID, '%1.2f\\%% & ', results(i, 5, 2));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(i, 6, 2));
     fprintf(fileID, ' \\addlinespace[3pt]\n');
end

%%%  WRITE AVERAGES for EU %%%
        avg_EU = sum(E_i(id_EU,:).*results(id_EU,:,2)) ...
                                                         ./sum(E_i(id_EU,:));
        %avg_EU(1) = mean(results(id_EU,1,1));
        
        fprintf(fileID, 'EU  & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_EU(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_EU(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_EU(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_EU(6));

        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_RoW = sum(E_i(id_RoW,:).*results(id_RoW,:,2))./sum(E_i(id_RoW,:));
        
        fprintf(fileID, 'RoW & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_RoW(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_RoW(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_RoW(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_RoW(6)); 


          mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{4}{l}{\textbf{Case 3: US trade war with China (108\% tariff)}} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
for i = [id_US, id_CHN]
    fprintf(fileID, '%s & ', country_names{i});
    fprintf(fileID, '%1.2f\\%% & ', results(i, 1, 3));
    fprintf(fileID, '%1.1f\\%% & ', results(i, 2, 3));
    fprintf(fileID, '%1.2f\\%% & ', results(i, 5, 3));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(i, 6, 3));
     fprintf(fileID, ' \\addlinespace[3pt]\n');
end

        avg_EU = sum(E_i(id_EU,:).*results(id_EU,:,3)) ...
                                                         ./sum(E_i(id_EU,:));
        %avg_non_US(1) = mean(results(id_EU,1,1));
        
        fprintf(fileID, 'EU & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_EU(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_EU(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_EU(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_EU(6));



        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_RoW = sum(E_i(id_RoW,:).*results(id_RoW,:,3))./sum(E_i(id_RoW,:));
        
        fprintf(fileID, 'RoW& ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_RoW(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_RoW(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_RoW(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_RoW(6)); 


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
%ERR1(N,1) = w_i_h(num_id) -1;  % replace one excess equation
% ------------------------------------------------------------------
%        Total Income = Total Sales 
% ------------------------------------------------------------------
%X_global = sum(sum(lambda_ji.*E_i.*(1- eye(N))),2);
%X_global_new = sum(sum(lambda_ji_new.*E_i_new.*(1- eye(N))),2);

X_global = sum(Y_i);
X_global_new = sum(Y_i_new);

ERR2 = tariff_rev + (w_i_h.*L_i_h.*Y_i)  + T_i.*(X_global_new./X_global)  -  E_i_new;

% ------------------------------------------------------------------

ERR3 = L_i_h - (tau_i_h.*w_i_h./P_i_h).^kappa;

ceq= [ERR1' ERR2' ERR3'];

delta_i = E_i./(E_i - kappa*(1-tau_i).*Y_i/(1+kappa));
W_i_h = delta_i .* (E_i_h ./ P_i_h) + (1-delta_i).*(w_i_h.*L_i_h ./ P_i_h);
%W_i_h = (E_i_h ./ P_i_h);

% factual trade flows
X_ji = lambda_ji.*repmat(E_i',N,1);
D_i =  sum(X_ji,1)' - sum(X_ji,2) ;
D_i_new =  sum(X_ji_new,1)' - sum(X_ji_new,2);


d_welfare = 100*(W_i_h-1);
d_export = 100*( sum(X_ji_new.*(1-eye(N)),2)./ sum(X_ji.*(1-eye(N)),2) - 1);
d_import = 100*( sum(X_ji_new.*(1-eye(N)),1)./ sum(X_ji.*(1-eye(N)),1) - 1)';
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
r_22 = (E_i(2) - X(1,2))/(E_i(2) - X(1,2)+ X(2,1));

F(1) = (1-r_11)*nu(2) +  r_11*nu(1) - 0.12;
F(2) = r_22*nu(2) +  (1-r_22)*nu(1) - 0.26;


end