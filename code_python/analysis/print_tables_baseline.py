"""
Print LaTeX tables for baseline model results.

Converts MATLAB print_tables_baseline.m to Python.
Generates Tables 1, 2, 3, and 9 from the paper.
"""

import os
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import get_output_dir


def print_table_1(results, revenue, E_i, country_names, id_US, output_dir):
    """
    Generate Table 1: Baseline tariff scenarios.

    Shows welfare, deficit, exports/GDP, imports/GDP, employment, and prices
    for three cases:
    - Case 1: USTR tariffs + income tax relief + no retaliation (scenario 0)
    - Case 2: USTR tariffs + lump-sum rebate + no retaliation (scenario 7)
    - Case 3: Optimal US tariffs + income tax relief + no retaliation (scenario 3)
    """

    filepath = os.path.join(output_dir, 'Table_1.tex')

    with open(filepath, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lcccccc}\n')
        f.write('\\toprule\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Case 1: USTR tariffs + income tax relief + no retaliation}}  \\\\\n')
        f.write('\\midrule\n')
        f.write('Country &\n')
        f.write('\\specialcell{$\\Delta$ welfare} &\n')
        f.write('\\specialcell{$\\Delta$ deficit} &\n')
        f.write('\\specialcell{$\\Delta$ $\\frac{\\textrm{exports}}{\\textrm{GDP}}$} & \n')
        f.write('\\specialcell{$\\Delta$ $\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &\n')
        f.write('\\specialcell{$\\Delta$ employment} &\n')
        f.write('\\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('\\midrule\n')

        # Case 1: scenario index 0
        scenario_idx = 0
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        non_US_mask = np.ones(len(E_i), dtype=bool)
        non_US_mask[id_US] = False
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Case 2: scenario index 7 (lump-sum rebate)
        f.write('\\midrule\n')
        f.write('\\addlinespace[10pt]\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Case 2: USTR tariffs + lump-sum rebate + no retaliation}} \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 7
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Case 3: scenario index 3 (optimal tariff)
        f.write('\\midrule\n')
        f.write('\\addlinespace[10pt]\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Case 3: optimal US tariffs + income tax relief + no retaliation}} \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 3
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Table closing
        f.write(' \\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f"  Table 1 saved to: {filepath}")


def print_table_2(results, E_i, country_names, id_US, id_CHN, id_EU, non_US, output_dir):
    """
    Generate Table 2: Retaliation scenarios.

    Shows welfare, deficit, employment, and prices for:
    - (1) USTR tariff + reciprocal retaliation (scenario 5)
    - (3) USTR tariff + optimal retaliation (scenario 4)
    - (4) Optimal tariff + optimal retaliation (scenario 6)
    """

    filepath = os.path.join(output_dir, 'Table_2.tex')

    with open(filepath, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lcccc}\n')
        f.write('\\toprule\n')
        f.write('\\multicolumn{5}{l}{\\textbf{(1) USTR tariff + reciprocal retaliation}} \\\\\n')
        f.write('\\midrule\n')
        f.write('Country &\n')
        f.write('$\\Delta$ welfare &\n')
        f.write('$\\Delta$ deficit &\n')
        f.write('$\\Delta$ employment &\n')
        f.write('$\\Delta$ real prices \\\\\n')
        f.write('\\midrule\n')

        # Case 1: scenario index 5 (reciprocal retaliation)
        scenario_idx = 5
        for i in [id_US, id_CHN]:
            f.write(f'{country_names[i]} & ')
            f.write(f'{results[i, 0, scenario_idx]:.2f}\\% & ')
            f.write(f'{results[i, 1, scenario_idx]:.1f}\\% & ')
            f.write(f'{results[i, 4, scenario_idx]:.2f}\\% & ')
            f.write(f'{results[i, 5, scenario_idx]:.1f}\\% \\\\ \n')
            f.write(' \\addlinespace[3pt]\n')

        # EU average
        E_i_EU = E_i[id_EU].reshape(-1, 1)  # Reshape for broadcasting
        avg_EU = np.sum(E_i_EU * results[id_EU, :, scenario_idx], axis=0) / np.sum(E_i[id_EU])
        f.write('EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\ \n')
        f.write(' \\addlinespace[3pt]\n')

        # Non-US average
        E_i_non_US = E_i[non_US].reshape(-1, 1)  # Reshape for broadcasting
        avg_RoW = np.sum(E_i_non_US * results[non_US, :, scenario_idx], axis=0) / np.sum(E_i[non_US])
        f.write('non-US (average) & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\ \n')

        # Case 3: scenario index 4 (optimal retaliation)
        f.write('\\bottomrule\n')
        f.write('\\addlinespace[15pt]\n')
        f.write('\\multicolumn{5}{l}{\\textbf{(3) USTR tariff + optimal retaliation}}  \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 4
        for i in [id_US, id_CHN]:
            f.write(f'{country_names[i]} & ')
            f.write(f'{results[i, 0, scenario_idx]:.2f}\\% & ')
            f.write(f'{results[i, 1, scenario_idx]:.1f}\\% & ')
            f.write(f'{results[i, 4, scenario_idx]:.2f}\\% & ')
            f.write(f'{results[i, 5, scenario_idx]:.1f}\\% \\\\ \n')
            f.write(' \\addlinespace[3pt]\n')

        # EU average
        E_i_EU = E_i[id_EU].reshape(-1, 1)  # Reshape for broadcasting
        avg_EU = np.sum(E_i_EU * results[id_EU, :, scenario_idx], axis=0) / np.sum(E_i[id_EU])
        f.write('EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\ \n')
        f.write(' \\addlinespace[3pt]\n')

        # Non-US average
        E_i_non_US = E_i[non_US].reshape(-1, 1)  # Reshape for broadcasting
        avg_RoW = np.sum(E_i_non_US * results[non_US, :, scenario_idx], axis=0) / np.sum(E_i[non_US])
        f.write('non-US (average) & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\ \n')

        # Case 4: scenario index 6 (optimal tariff + optimal retaliation)
        f.write('\\bottomrule\n')
        f.write('\\addlinespace[15pt]\n')
        f.write('\\multicolumn{5}{l}{\\textbf{(4) optimal tariff + optimal retaliation}}  \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 6
        for i in [id_US, id_CHN]:
            f.write(f'{country_names[i]} & ')
            f.write(f'{results[i, 0, scenario_idx]:.2f}\\% & ')
            f.write(f'{results[i, 1, scenario_idx]:.1f}\\% & ')
            f.write(f'{results[i, 4, scenario_idx]:.2f}\\% & ')
            f.write(f'{results[i, 5, scenario_idx]:.1f}\\% \\\\ \n')
            f.write(' \\addlinespace[3pt]\n')

        # EU average
        E_i_EU = E_i[id_EU].reshape(-1, 1)  # Reshape for broadcasting
        avg_EU = np.sum(E_i_EU * results[id_EU, :, scenario_idx], axis=0) / np.sum(E_i[id_EU])
        f.write('EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\ \n')
        f.write(' \\addlinespace[3pt]\n')

        # Non-US average
        E_i_non_US = E_i[non_US].reshape(-1, 1)  # Reshape for broadcasting
        avg_RoW = np.sum(E_i_non_US * results[non_US, :, scenario_idx], axis=0) / np.sum(E_i[non_US])
        f.write('non-US (average) & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\ \n')

        # Table closing
        f.write(' \\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f"  Table 2 saved to: {filepath}")


def print_table_3(revenue, output_dir):
    """
    Generate Table 3: Tariff revenue.

    Shows revenue as % of GDP and % of Federal Budget for:
    - USTR tariff (scenario 0)
    - Optimal tariff (scenario 3)
    - Optimal retaliation (scenario 4)
    - Reciprocal retaliation (scenario 5)
    """

    filepath = os.path.join(output_dir, 'Table_3.tex')

    with open(filepath, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lcccc}\n')
        f.write('\\toprule\n')
        f.write('& & & \\multicolumn{2}{c}{retaliation} \\\\\n')
        f.write('\\cmidrule(lr){4-5}\n')
        f.write('& USTR tariff & optimal tariff &  optimal & reciprocal \\\\\n')
        f.write('\\midrule\n')

        # Revenue as % of GDP
        f.write('\\% of GDP &')
        f.write(f'{100*revenue[0]:.2f}\\% & ')
        f.write(f'{100*revenue[3]:.2f}\\% &')
        f.write(f'{100*revenue[4]:.2f}\\% & ')
        f.write(f'{100*revenue[5]:.2f}\\% \\\\ \n')

        # Revenue as % of Federal Budget (federal budget is 23% of GDP)
        f.write('\\% of Federal Budget &')
        f.write(f'{100*revenue[0]/0.23:.2f}\\% & ')
        f.write(f'{100*revenue[3]/0.23:.2f}\\% &')
        f.write(f'{100*revenue[4]/0.23:.2f}\\% & ')
        f.write(f'{100*revenue[5]/0.23:.2f}\\% \\\\ \n')

        # Table closing
        f.write(' \\addlinespace[3pt]\n')
        f.write(' \\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f"  Table 3 saved to: {filepath}")


def print_table_9(results, E_i, country_names, id_US, output_dir):
    """
    Generate Table 9: Model variants (Appendix).

    Shows welfare, deficit, exports/GDP, imports/GDP, employment, and prices for:
    - Baseline model
    - Alternative 2: incomplete passthrough (scenario 1)
    - Alternative 3: higher trade elasticity (scenario 8)
    - Alternative 4: Eaton-Kortum-Krugman model (scenario 2)

    Note: Multi-sector alternative (Alternative 1) requires IO model which is not yet working.
    """

    filepath = os.path.join(output_dir, 'Table_9.tex')

    with open(filepath, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lcccccc}\n')
        f.write('\\toprule\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Baseline model:($\\tilde{\\varphi}=1$, $\\varphi>1$, $\\varepsilon=4$)}}  \\\\\n')
        f.write('\\midrule\n')
        f.write('Country &\n')
        f.write('\\specialcell{$\\Delta$ welfare} &\n')
        f.write('\\specialcell{$\\Delta$ deficit} &\n')
        f.write('\\specialcell{$\\Delta$ $\\frac{\\textrm{exports}}{\\textrm{GDP}}$} & \n')
        f.write('\\specialcell{$\\Delta$ $\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &\n')
        f.write('\\specialcell{$\\Delta$ employment} &\n')
        f.write('\\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('\\midrule\n')

        # Baseline: scenario index 0
        scenario_idx = 0
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        non_US_mask = np.ones(len(E_i), dtype=bool)
        non_US_mask[id_US] = False
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # NOTE: Multi-sector alternative would go here but requires IO model
        # Skipping for now as IO model has convergence issues

        # Alternative 2: incomplete passthrough, scenario index 1
        f.write('\\midrule\n')
        f.write('\\addlinespace[10pt]\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Alternative 2: incomplete passthrough to firm-level prices ($\\tilde{\\varphi}=0.25$)}}  \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 1
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Alternative 3: higher trade elasticity, scenario index 8
        f.write('\\midrule\n')
        f.write('\\addlinespace[10pt]\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Alternative 3: higher trade elasticity ($\\varepsilon=8$)}}  \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 8
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Alternative 4: Eaton-Kortum-Krugman, scenario index 2
        f.write('\\midrule\n')
        f.write('\\addlinespace[10pt]\n')
        f.write('\\multicolumn{7}{l}{\\textbf{Alternative 4: Eaton-Kortum-Krugman model ($\\varphi=1$, $\\nu=0$)}} \\\\ \n')
        f.write('\\midrule\n')

        scenario_idx = 2
        f.write(f'{country_names[id_US]} & ')
        f.write(f'{results[id_US, 0, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 1, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 2, scenario_idx]:.1f}\\% &')
        f.write(f'{results[id_US, 3, scenario_idx]:.1f}\\% & ')
        f.write(f'{results[id_US, 4, scenario_idx]:.2f}\\% & ')
        f.write(f'{results[id_US, 5, scenario_idx]:.1f}\\% \\\\ \n')

        # Write averages for non-US
        f.write(' \\addlinespace[3pt]\n')
        E_i_non_US = E_i[non_US_mask].reshape(-1, 1)  # Reshape for broadcasting
        avg_non_US = np.sum(E_i_non_US * results[non_US_mask, :, scenario_idx], axis=0) / np.sum(E_i[non_US_mask])

        f.write('non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\ \n')

        # Table closing
        f.write(' \\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f"  Table 9 saved to: {filepath}")


def generate_all_tables(results_dict, base_path='.'):
    """
    Generate all LaTeX tables from baseline model results.

    Parameters:
    -----------
    results_dict : dict
        Dictionary returned by main_baseline.main() containing all results
    base_path : str
        Base path for output directory
    """

    print("\nGenerating LaTeX tables...")

    # Extract data from results dictionary
    results = results_dict['results']
    revenue = results_dict['revenue']
    E_i = results_dict['E_i']
    country_names = results_dict['country_names']
    id_US = results_dict['id_US']
    id_CHN = results_dict['id_CHN']
    id_EU = results_dict['id_EU']
    non_US = results_dict['non_US']

    # Get output directory
    output_dir_name = get_output_dir()
    output_dir = os.path.join(base_path, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Generate each table
    print_table_1(results, revenue, E_i, country_names, id_US, output_dir)
    print_table_2(results, E_i, country_names, id_US, id_CHN, id_EU, non_US, output_dir)
    print_table_3(revenue, output_dir)
    print_table_9(results, E_i, country_names, id_US, output_dir)

    print("\n[OK] All baseline LaTeX tables generated successfully!")


if __name__ == '__main__':
    # For testing, import and run baseline model
    from main_baseline import main

    print("Running baseline model to generate results...")
    results_dict = main()

    if results_dict:
        generate_all_tables(results_dict, base_path='../..')
    else:
        print("Error: Baseline model did not return results")
