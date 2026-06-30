
% This script produces the Tables 1-4 and 8-11 in 
% "Making America Great Again? The Economic Impacts of Liberation Day Tariffs" 
% by Anna Ignatenko, Luca Macedoni, Ahmad Lashkaripour, Ina Simonovska
% for publication in the Journal of International Economics (Last update: July 2025)
% email for inquiries: ahmadlp@gmail.com

clc
clear all

% ------ Set Path -------
current_dir = pwd;
cd(current_dir)

%---- baseline model ----
run code/analysis/main_baseline.m             


%---- baseline + IO ----
run code/analysis/main_io.m  

%----- create fig 1 in Python ---
!python3 code/global_map/create_figure_1.py

%---- regional trade war ----
run code/analysis/main_regional.m

%---- alternative deficit framework ----
run code/analysis/main_deficit.m

