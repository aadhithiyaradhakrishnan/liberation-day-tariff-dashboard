% --------------------------------------------------------------------------------------
   %                 Print Output (Table 1)
% --------------------------------------------------------------------------------------

tablePreamble = {...
'\begin{tabular}{lcccccccc}';
        '\toprule';
        '\multicolumn{6}{l}{\textbf{Case 1: USTR tariffs + income tax relief + no retlaiation}}  \\';
        '\midrule';
        'Country &';
        '\specialcell{$\Delta$ welfare} &';
        '\specialcell{$\Delta$ deficit &';
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

   fileID = fopen('../../output/Table_1.tex', 'w');

%%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end

%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 1));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 1));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 1));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 1));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 1));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 1));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,1)) ...
                                                         ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,1));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));



        mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{Case 2: USTR tariffs + lump-sum rebate + no retlaiation}} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 8));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 8));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 8));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 8));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 8));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 8));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,8)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,3));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));


        mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{Case 3: optimal US tariffs + income tax relief + no retlaiation}} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 4));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 4));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 4));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 4));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 4));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 4));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,4)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,4));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));        

%%% TABLE CLOSING  %%%
for n = 1:numel(tableClosing)
    fprintf(fileID, '%s\n', tableClosing{n});
end

fclose(fileID);



% --------------------------------------------------------------------------------------
   %                 Print Output (Table 2)
% --------------------------------------------------------------------------------------

tablePreamble = {...
'\begin{tabular}{lccccc}';
        '\toprule';
        '\multicolumn{5}{l}{\textbf{(1) USTR tariff + reciprocal retaliation}} \\';
        '\midrule';
        'Country &';
        '$\Delta$ welfare &';
        '$\Delta$ deficit &';
        '$\Delta$ employment &';
        '$\Delta$ real prices \\';

        '\midrule'
        };

   tableClosing = {...
        ' \bottomrule';
        '\end{tabular}'
        };

   fileID = fopen('../../output/Table_2.tex', 'w');

%%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end


for i = [id_US, id_CHN]
    fprintf(fileID, '%s & ', country_names{i});
    fprintf(fileID, '%1.2f\\%% & ', results(i, 1, 6));
    fprintf(fileID, '%1.1f\\%% & ', results(i, 2, 6));
    fprintf(fileID, '%1.2f\\%% & ', results(i, 5, 6));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(i, 6, 6));
    fprintf(fileID, ' \\addlinespace[3pt]\n');
end    

%%%  WRITE AVERAGES %%%

        avg_EU = sum(E_i(id_EU,:).*results(id_EU,:,6))./sum(E_i(id_EU,:));                                             
        
        fprintf(fileID, 'EU & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_EU(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_EU(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_EU(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_EU(6));

        fprintf(fileID, ' \\addlinespace[3pt]\n');
        %avg_RoW = sum(E_i(id_RoW,:).*results(id_RoW,:,6))./sum(E_i(id_RoW,:));
        avg_RoW = sum(E_i(non_US,:).*results(non_US,:,6))./sum(E_i(non_US,:)); 
        
        %fprintf(fileID, 'RoW & ');
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_RoW(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_RoW(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_RoW(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_RoW(6));

        mid_table_3 = {...            
        '\bottomrule';
        '\addlinespace[15pt]';
        '\multicolumn{5}{l}{\textbf{(3) USTR tariff + optimal retaliation}}  \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table_3)
       fprintf(fileID, '%s\n', mid_table_3{n});
       end


for i = [id_US, id_CHN]
    fprintf(fileID, '%s & ', country_names{i});
    fprintf(fileID, '%1.2f\\%% & ', results(i, 1, 5));
    fprintf(fileID, '%1.1f\\%% & ', results(i, 2, 5));
    fprintf(fileID, '%1.2f\\%% & ', results(i, 5, 5));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(i, 6, 5));
    fprintf(fileID, ' \\addlinespace[3pt]\n');
end    

%%%  WRITE AVERAGES %%%


        avg_EU = sum(E_i(id_EU,:).*results(id_EU,:,5))./sum(E_i(id_EU,:));                                             
        
        fprintf(fileID, 'EU & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_EU(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_EU(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_EU(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_EU(6));

        fprintf(fileID, ' \\addlinespace[3pt]\n');
        %avg_RoW = sum(E_i(id_RoW,:).*results(id_RoW,:,5))./sum(E_i(id_RoW,:));
        avg_RoW = sum(E_i(non_US,:).*results(non_US,:,5))./sum(E_i(non_US,:));               
        
        %fprintf(fileID, 'RoW & ');
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_RoW(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_RoW(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_RoW(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_RoW(6));
        
    mid_table_4 = {...            
        '\bottomrule';
        '\addlinespace[15pt]';
        '\multicolumn{5}{l}{\textbf{(4) optimal tariff + optimal retaliation}}  \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table_4)
       fprintf(fileID, '%s\n', mid_table_4{n});
       end


for i = [id_US, id_CHN]
    fprintf(fileID, '%s & ', country_names{i});
    fprintf(fileID, '%1.2f\\%% & ', results(i, 1, 7));
    fprintf(fileID, '%1.1f\\%% & ', results(i, 2, 7));
    fprintf(fileID, '%1.2f\\%% & ', results(i, 5, 7));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(i, 6, 7));
    fprintf(fileID, ' \\addlinespace[3pt]\n');
end    

%%%  WRITE AVERAGES %%%
        

        avg_EU = sum(E_i(id_EU,:).*results(id_EU,:,7))./sum(E_i(id_EU,:));                                             
        
        fprintf(fileID, 'EU & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_EU(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_EU(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_EU(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_EU(6));

        fprintf(fileID, ' \\addlinespace[3pt]\n');
        %avg_RoW = sum(E_i(id_RoW,:).*results(id_RoW,:,7))./sum(E_i(id_RoW,:));
        avg_RoW = sum(E_i(non_US,:).*results(non_US,:,7))./sum(E_i(non_US,:)); 
        
        %fprintf(fileID, 'RoW & ');
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_RoW(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_RoW(2));
        fprintf(fileID, '%1.2f\\%% & ', avg_RoW(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_RoW(6));  

%%% TABLE CLOSING  %%%
for n = 1:numel(tableClosing)
    fprintf(fileID, '%s\n', tableClosing{n});
end

fclose(fileID);


% --------------------------------------------------------------------------------------
   %                 Print Output (Table 3)
% --------------------------------------------------------------------------------------

tablePreamble = {...
'\begin{tabular}{lccccc}';
        '\toprule';
        '& & & \multicolumn{2}{c}{retaliation} \\';
        '\cmidrule(lr){4-5}';
        '& USTR tariff$ & optimal tariff &  optimal & reciprocal \\'
        '\midrule'
        };

   tableClosing = {...
        ' \bottomrule';
        '\end{tabular}'
        };

   fileID = fopen('../../output/Table_3.tex', 'w');

   %%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end
    
    fprintf(fileID, '\\%% of GDP &');
    fprintf(fileID, '%1.2f\\%% & ', 100*revenue(1));
    fprintf(fileID, '%1.2f\\%% &', 100*revenue(4));
    fprintf(fileID, '%1.2f\\%% & ', 100*revenue(5));
    fprintf(fileID, '%1.2f\\%% \\\\ ', 100*revenue(6));

    fprintf(fileID, '\\%% of Federal Budget &');
    fprintf(fileID, '%1.2f\\%% & ', 100*revenue(1)/0.23);
    fprintf(fileID, '%1.2f\\%% &', 100*revenue(4)/0.23);
    fprintf(fileID, '%1.2f\\%% & ', 100*revenue(5)/0.23);
    fprintf(fileID, '%1.2f\\%% \\\\ ', 100*revenue(6)/0.23);

     fprintf(fileID, ' \\addlinespace[3pt]\n');
    for n = 1:numel(tableClosing)
    fprintf(fileID, '%s\n', tableClosing{n});
end

fclose(fileID);


% --------------------------------------------------------------------------------------
   %                 Print Output (Table 9, Appendix)
% --------------------------------------------------------------------------------------

tablePreamble = {...
'\begin{tabular}{lcccccccc}';
        '\toprule';
        '\multicolumn{6}{l}{\textbf{Baseline model:($\tilde{\varphi}=1$, $\varphi>1$, $\varepsilon=4$)}}  \\';
        '\midrule';
        'Country &';
        '\specialcell{$\Delta$ welfare} &';
        '\specialcell{$\Delta$ deficit} &';
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

   fileID = fopen('../../output/Table_9.tex', 'w');

%%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end

%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 1));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 1));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 1));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 1));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 1));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 1));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,1)) ...
                                                         ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,1));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));

   mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{Alternative 1: multiple sectors}}  \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results_multi(id_US_new, 1));
    fprintf(fileID, '%1.1f\\%% & ', results_multi(id_US_new, 2));
    fprintf(fileID, '%1.1f\\%% &', results_multi(id_US_new, 3));
    fprintf(fileID, '%1.1f\\%% & ', results_multi(id_US_new, 4));
    fprintf(fileID, '%1.2f\\%% & ', results_multi(id_US_new, 5));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results_multi(id_US_new, 6,1));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i_multi([1:id_US_new-1, id_US_new+1:end],:).*results_multi([1:id_US_new-1, id_US_new+1:end],:,1)) ...
                                                        ./sum(E_i_multi([1:id_US_new-1, id_US_new+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,2));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));



        mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{Alternative 2: incomplete passthrough to firm-level prices ($\tilde{\varphi}=0.25$)}}  \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 2));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 2));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 2));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 2));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 2));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 2));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,2)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,2));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));

        mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{Alternative 3: higher trade elasticity ($\varepsilon=8$)}}  \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 9));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 9));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 9));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 9));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 9));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 9));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,9)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,2));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));

        

          mid_table = {...            
        '\midrule';
        '\addlinespace[10pt]';
        '\multicolumn{6}{l}{\textbf{Alternative 4: Eaton-Kortum-Krugman model ($\varphi=1$, $\nu=0$)}} \\ ';
        '\midrule';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 3));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 3));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3, 3));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 3));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 3));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 3));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,3)) ...
                                                        ./sum(E_i([1:id_US-1, id_US+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,3));
        
        fprintf(fileID, 'non-US (average) & ');
        fprintf(fileID, '%1.2f\\%%  & ', avg_non_US(1));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(2));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(3));
        fprintf(fileID, '%1.1f\\%% & ', avg_non_US(4));
        fprintf(fileID, '%1.2f\\%% & ', avg_non_US(5));
        fprintf(fileID, '%1.1f\\%% \\\\ ', avg_non_US(6));

%%% TABLE CLOSING  %%%
for n = 1:numel(tableClosing)
    fprintf(fileID, '%s\n', tableClosing{n});
end

fclose(fileID);

