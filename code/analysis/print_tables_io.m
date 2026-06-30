% --------------------------------------------------------------------------------------
   %                 Print Output (Table 4)
% --------------------------------------------------------------------------------------
countries = readtable('../../data/base_data/country_labels.csv');
country_names = countries.iso3; % Adjust 'Country' to match actual column name in CSV

tablePreamble = {...
'\begin{tabular}{lrrrrrr}';
        '\toprule';
        '&';
        '\specialcell{$\Delta$ welfare} &';
        '\specialcell{$\Delta$ deficit} &';
        '\specialcell{$\Delta$$\frac{\textrm{exports}}{\textrm{GDP}}$} & ';
        '\specialcell{$\Delta$$\frac{\textrm{imports}}{\textrm{GDP}}$} &';
        '\specialcell{$\Delta$ emp} &';
        '\specialcell{$\Delta$ prices} \\';
        '\addlinespace[-8pt]';
        '\multicolumn{7}{l}{\textbf{Pre-Retaliation Scenarios}} \\';
        '\midrule';
        '\addlinespace[5pt]';
        '\textbf{(1)} \textit{USTR tariffs + one sector} \\';
        '\cmidrule(lr){1-1}';
        '\addlinespace[3pt]';

        };

   tableClosing = {...
        ' \bottomrule';
        '\end{tabular}'
        };

   fileID = fopen('../../output/Table_4.tex', 'w');

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
        avg_non_US = sum(Y_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,1)) ...
                                                         ./sum(Y_i([1:id_US-1, id_US+1:end],:));
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
        '\addlinespace[10pt]'
        '\textbf{(2)}  \textit{Optimal tariff + one sector} \\';
        '\cmidrule(lr){1-1}';
        '\addlinespace[3pt]';
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
        avg_non_US = sum(Y_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,2)) ...
                                                        ./sum(Y_i([1:id_US-1, id_US+1:end],:));
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
        '\addlinespace[10pt]'
        '\textbf{(3)}  \textit{USTR tariffs + multiple sectors} \\';
        '\cmidrule(lr){1-1}';
        '\addlinespace[3pt]';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results_multi(id_US_new, 1,1));
    fprintf(fileID, '%1.1f\\%% & ', results_multi(id_US_new, 2,1));
    fprintf(fileID, '%1.1f\\%% &', results_multi(id_US_new, 3,1));
    fprintf(fileID, '%1.1f\\%% & ', results_multi(id_US_new, 4,1));
    fprintf(fileID, '%1.2f\\%% & ', results_multi(id_US_new, 5,1));
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
        '\addlinespace[10pt]'
        '\multicolumn{7}{l}{\textbf{Post-Retaliation Scenarios}} \\';
        '\midrule';
        '\addlinespace[5pt]'
        '\textbf{(1)}  \textit{reciprocal retaliation + one sector} \\';
        '\cmidrule(lr){1-1}';
        '\addlinespace[3pt]';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 1, 3));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 2, 3));
    fprintf(fileID, '%1.1f\\%% &', results(id_US, 3,  3));
    fprintf(fileID, '%1.1f\\%% & ', results(id_US, 4, 3));
    fprintf(fileID, '%1.2f\\%% & ', results(id_US, 5, 3));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results(id_US, 6, 3));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(Y_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,3)) ...
                                                        ./sum(Y_i([1:id_US-1, id_US+1:end],:));
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
        '\addlinespace[5pt]'
        '\textbf{(2)}  \textit{optimal retaliation + one sector} \\';
        '\cmidrule(lr){1-1}';
        '\addlinespace[3pt]';
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
        avg_non_US = sum(Y_i([1:id_US-1, id_US+1:end],:).*results([1:id_US-1, id_US+1:end],:,4)) ...
                                                        ./sum(Y_i([1:id_US-1, id_US+1:end],:));
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
        '\addlinespace[5pt]'
        '\textbf{(3)}  \textit{reciprocal retaliation + multiple sectors}  \\';
        '\cmidrule(lr){1-1}';
        };

       for n = 1:numel(mid_table)
       fprintf(fileID, '%s\n', mid_table{n});
       end


%%%  COLUMNS WITH RESULTS %%%
    fprintf(fileID, '%s & ', country_names{id_US});
    fprintf(fileID, '%1.2f\\%% & ', results_multi(id_US_new, 1,2));
    fprintf(fileID, '%1.1f\\%% & ', results_multi(id_US_new, 2,2));
    fprintf(fileID, '%1.1f\\%% &', results_multi(id_US_new, 3,2));
    fprintf(fileID, '%1.1f\\%% & ', results_multi(id_US_new, 4,2));
    fprintf(fileID, '%1.2f\\%% & ', results_multi(id_US_new, 5,2));
    fprintf(fileID, '%1.1f\\%% \\\\ ', results_multi(id_US_new, 6,2));

%%%  WRITE AVERAGES %%%
        fprintf(fileID, ' \\addlinespace[3pt]\n');
        avg_non_US = sum(E_i_multi([1:id_US_new-1, id_US_new+1:end],:).*results_multi([1:id_US_new-1, id_US_new+1:end],:,2)) ...
                                                        ./sum(E_i_multi([1:id_US_new-1, id_US_new+1:end],:));
        %avg_non_US(1) = mean(results([1:id_US-1, id_US+1:end],1,2));
        
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
   %                 Print Output (Table 11 -- Appendix)
% --------------------------------------------------------------------------------------

tablePreamble = {...
'\begin{tabular}{lcccccccccc}';
        '\toprule';
        '& \multicolumn{4}{c}{before retaliation} && \multicolumn{4}{c}{after retaliation} \\';
        '\cmidrule(lr){2-5} \cmidrule(lr){7-10}';
        'main & IO & multi & multi + IO  && main & IO & multi & multi + IO \\'
        '\midrule'
        };

   tableClosing = {...
        ' \bottomrule';
        '\end{tabular}'
        };

   fileID = fopen('../../output/Table_11.tex', 'w');

   %%% TABLE PREAMBLE   %%%
for n = 1:numel(tablePreamble)
    fprintf(fileID, '%s\n', tablePreamble{n});
end
    
    fprintf(fileID, '$\\Delta$ global trade-to-GDP &');
    fprintf(fileID, '%1.1f\\%% & ', d_trade(1));
    fprintf(fileID, '%1.1f\\%% &', d_trade_IO(1));
    fprintf(fileID, '%1.1f\\%% & ', d_trade(8));
    fprintf(fileID, '%1.1f\\%% && ', d_trade_IO_multi(1));
    fprintf(fileID, '%1.1f\\%% & ', d_trade(6));
    fprintf(fileID, '%1.1f\\%% &', d_trade_IO(2));
    fprintf(fileID, '%1.1f\\%% & ', d_trade(9));
    fprintf(fileID, '%1.1f\\%% \\\\ ', d_trade_IO_multi(2));

    fprintf(fileID, ' \\addlinespace[3pt]\n');

    fprintf(fileID, '$\\Delta$ global employment &');
    fprintf(fileID, '%1.2f\\%% & ', d_employment(1));
    fprintf(fileID, '%1.2f\\%% &', d_employment_IO(1));
    fprintf(fileID, '%1.2f\\%% & ', d_employment(8));
    fprintf(fileID, '%1.2f\\%% && ', d_employment_IO_multi(1));
    fprintf(fileID, '%1.2f\\%% & ', d_employment(6));
    fprintf(fileID, '%1.2f\\%% &', d_employment_IO(2));
    fprintf(fileID, '%1.2f\\%% & ', d_employment(9));
    fprintf(fileID, '%1.2f\\%% \\\\ ', d_employment_IO_multi(2));

     fprintf(fileID, ' \\addlinespace[3pt]\n');
    for n = 1:numel(tableClosing)
    fprintf(fileID, '%s\n', tableClosing{n});
    end

fclose(fileID);

delete ../../output/Table_11.mat
