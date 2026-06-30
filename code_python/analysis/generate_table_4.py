"""
Generate Table 4: One-Sector vs Multi-Sector Comparison
Combines results from baseline single-sector and multi-sector models
"""

import numpy as np
import pandas as pd
import os
import sys

def generate_table_4():
    """Generate Table 4 LaTeX output"""
    print("=" * 80)
    print("Generating Table 4: One-Sector vs Multi-Sector Comparison")
    print("=" * 80)

    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = os.path.join(base_path, 'python_output')

    # Load baseline single-sector results
    print("\nLoading baseline single-sector results...")
    baseline_data = np.load(os.path.join(output_dir, 'baseline_results.npz'))
    results = baseline_data['results']  # Shape: (N, 7, 10) for N countries, 7 metrics, 10 scenarios
    Y_i = baseline_data['Y_i']
    id_US = baseline_data['id_US']

    # Load country names
    country_names = pd.read_csv(os.path.join(base_path, 'data', 'base_data', 'country_labels.csv'))['iso3'].values

    # Load multi-sector IO results (MATLAB's print_tables_io.m uses sub_multisector_io results for Table 4)
    print("Loading multi-sector IO results...")
    multi_data = np.load(os.path.join(output_dir, 'multisector_io_results.npz'))
    results_multi = multi_data['results_multi']  # Shape: (N_filtered, 7, 2) for filtered countries
    id_US_multi = multi_data['id_US']

    # Load multi-sector E_i for averaging
    # We need to reload the filtered E_i from multi-sector model
    data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
    trade_data = pd.read_csv(data_path, header=None)
    X = trade_data.iloc[:, 3].values
    N_orig = 194
    K = 4
    X_ji = X.reshape((N_orig, N_orig, K), order='F')

    problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
    ID = np.where(problematic_id == 1)[0]
    idx = np.setdiff1d(np.arange(N_orig), ID)
    N_multi = len(idx)

    X_ji_filtered = np.zeros((N_multi, N_multi, K))
    for k in range(K):
        X_ji_filtered[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N_multi, N_multi)
    E_i_multi = np.sum(np.sum(X_ji_filtered, axis=0), axis=1)

    print(f"Single-sector: N={len(Y_i)}, id_US={id_US}")
    print(f"Multi-sector: N={N_multi}, id_US_multi={id_US_multi}")

    # Scenario mapping:
    # Single-sector: scenario 0 = USTR no retaliation, scenario 3 = optimal, scenario 5 = reciprocal retal, scenario 4 = optimal retal
    # Multi-sector: scenario 0 = USTR no retaliation, scenario 1 = reciprocal retaliation

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")

    lines = []
    lines.append('\\begin{tabular}{lrrrrrr}')
    lines.append('\\toprule')
    lines.append('&')
    lines.append('\\specialcell{$\\Delta$ welfare} &')
    lines.append('\\specialcell{$\\Delta$ deficit} &')
    lines.append('\\specialcell{$\\Delta$$\\frac{\\textrm{exports}}{\\textrm{GDP}}$} &')
    lines.append('\\specialcell{$\\Delta$$\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &')
    lines.append('\\specialcell{$\\Delta$ emp} &')
    lines.append('\\specialcell{$\\Delta$ prices} \\\\')
    lines.append('\\addlinespace[-8pt]')
    lines.append('\\multicolumn{7}{l}{\\textbf{Pre-Retaliation Scenarios}} \\\\')
    lines.append('\\midrule')
    lines.append('\\addlinespace[5pt]')
    lines.append('\\textbf{(1)} \\textit{USTR tariffs + one sector} \\\\')
    lines.append('\\cmidrule(lr){1-1}')
    lines.append('\\addlinespace[3pt]')

    # (1) USTR tariffs + one sector (scenario 0)
    s = 0
    lines.append(f'{country_names[id_US]} & ')
    lines.append(f'{results[id_US, 0, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 1, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 2, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 3, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 4, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 5, s]:.1f}\\% \\\\')

    # Non-US average
    lines.append('\\addlinespace[3pt]')
    non_US = np.arange(len(Y_i)) != id_US
    Y_non_US = Y_i[non_US].reshape(-1, 1)
    avg_non_US = np.sum(Y_non_US * results[non_US, :, s], axis=0) / np.sum(Y_i[non_US])
    lines.append(f'non-US (average) & ')
    lines.append(f'{avg_non_US[0]:.2f}\\% & ')
    lines.append(f'{avg_non_US[1]:.1f}\\% & ')
    lines.append(f'{avg_non_US[2]:.1f}\\% & ')
    lines.append(f'{avg_non_US[3]:.1f}\\% & ')
    lines.append(f'{avg_non_US[4]:.2f}\\% & ')
    lines.append(f'{avg_non_US[5]:.1f}\\% \\\\')

    # (2) Optimal tariff + one sector (scenario 3)
    lines.append('\\midrule')
    lines.append('\\addlinespace[10pt]')
    lines.append('\\textbf{(2)} \\textit{Optimal tariff + one sector} \\\\')
    lines.append('\\cmidrule(lr){1-1}')
    lines.append('\\addlinespace[3pt]')

    s = 3
    lines.append(f'{country_names[id_US]} & ')
    lines.append(f'{results[id_US, 0, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 1, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 2, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 3, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 4, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 5, s]:.1f}\\% \\\\')

    lines.append('\\addlinespace[3pt]')
    avg_non_US = np.sum(Y_non_US * results[non_US, :, s], axis=0) / np.sum(Y_i[non_US])
    lines.append(f'non-US (average) & ')
    lines.append(f'{avg_non_US[0]:.2f}\\% & ')
    lines.append(f'{avg_non_US[1]:.1f}\\% & ')
    lines.append(f'{avg_non_US[2]:.1f}\\% & ')
    lines.append(f'{avg_non_US[3]:.1f}\\% & ')
    lines.append(f'{avg_non_US[4]:.2f}\\% & ')
    lines.append(f'{avg_non_US[5]:.1f}\\% \\\\')

    # (3) USTR tariffs + multiple sectors (multi-sector scenario 0)
    lines.append('\\midrule')
    lines.append('\\addlinespace[10pt]')
    lines.append('\\textbf{(3)} \\textit{USTR tariffs + multiple sectors} \\\\')
    lines.append('\\cmidrule(lr){1-1}')
    lines.append('\\addlinespace[3pt]')

    s = 0
    lines.append(f'{country_names[id_US]} & ')
    lines.append(f'{results_multi[id_US_multi, 0, s]:.2f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 1, s]:.1f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 2, s]:.1f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 3, s]:.1f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 4, s]:.2f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 5, s]:.1f}\\% \\\\')

    lines.append('\\addlinespace[3pt]')
    non_US_multi = np.arange(N_multi) != id_US_multi
    E_non_US = E_i_multi[non_US_multi].reshape(-1, 1)
    avg_non_US = np.sum(E_non_US * results_multi[non_US_multi, :, s], axis=0) / np.sum(E_i_multi[non_US_multi])
    lines.append(f'non-US (average) & ')
    lines.append(f'{avg_non_US[0]:.2f}\\% & ')
    lines.append(f'{avg_non_US[1]:.1f}\\% & ')
    lines.append(f'{avg_non_US[2]:.1f}\\% & ')
    lines.append(f'{avg_non_US[3]:.1f}\\% & ')
    lines.append(f'{avg_non_US[4]:.2f}\\% & ')
    lines.append(f'{avg_non_US[5]:.1f}\\% \\\\')

    # Post-Retaliation Scenarios
    lines.append('\\midrule')
    lines.append('\\addlinespace[10pt]')
    lines.append('\\multicolumn{7}{l}{\\textbf{Post-Retaliation Scenarios}} \\\\')
    lines.append('\\midrule')
    lines.append('\\addlinespace[5pt]')
    lines.append('\\textbf{(1)} \\textit{reciprocal retaliation + one sector} \\\\')
    lines.append('\\cmidrule(lr){1-1}')
    lines.append('\\addlinespace[3pt]')

    # (1) Reciprocal retaliation + one sector (scenario 5)
    s = 5
    lines.append(f'{country_names[id_US]} & ')
    lines.append(f'{results[id_US, 0, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 1, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 2, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 3, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 4, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 5, s]:.1f}\\% \\\\')

    lines.append('\\addlinespace[3pt]')
    avg_non_US = np.sum(Y_non_US * results[non_US, :, s], axis=0) / np.sum(Y_i[non_US])
    lines.append(f'non-US (average) & ')
    lines.append(f'{avg_non_US[0]:.2f}\\% & ')
    lines.append(f'{avg_non_US[1]:.1f}\\% & ')
    lines.append(f'{avg_non_US[2]:.1f}\\% & ')
    lines.append(f'{avg_non_US[3]:.1f}\\% & ')
    lines.append(f'{avg_non_US[4]:.2f}\\% & ')
    lines.append(f'{avg_non_US[5]:.1f}\\% \\\\')

    # (2) Optimal retaliation + one sector (scenario 4)
    lines.append('\\midrule')
    lines.append('\\addlinespace[5pt]')
    lines.append('\\textbf{(2)} \\textit{optimal retaliation + one sector} \\\\')
    lines.append('\\cmidrule(lr){1-1}')
    lines.append('\\addlinespace[3pt]')

    s = 4
    lines.append(f'{country_names[id_US]} & ')
    lines.append(f'{results[id_US, 0, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 1, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 2, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 3, s]:.1f}\\% & ')
    lines.append(f'{results[id_US, 4, s]:.2f}\\% & ')
    lines.append(f'{results[id_US, 5, s]:.1f}\\% \\\\')

    lines.append('\\addlinespace[3pt]')
    avg_non_US = np.sum(Y_non_US * results[non_US, :, s], axis=0) / np.sum(Y_i[non_US])
    lines.append(f'non-US (average) & ')
    lines.append(f'{avg_non_US[0]:.2f}\\% & ')
    lines.append(f'{avg_non_US[1]:.1f}\\% & ')
    lines.append(f'{avg_non_US[2]:.1f}\\% & ')
    lines.append(f'{avg_non_US[3]:.1f}\\% & ')
    lines.append(f'{avg_non_US[4]:.2f}\\% & ')
    lines.append(f'{avg_non_US[5]:.1f}\\% \\\\')

    # (3) Reciprocal retaliation + multiple sectors (multi-sector scenario 1)
    lines.append('\\midrule')
    lines.append('\\addlinespace[5pt]')
    lines.append('\\textbf{(3)} \\textit{reciprocal retaliation + multiple sectors} \\\\')
    lines.append('\\cmidrule(lr){1-1}')

    s = 1
    lines.append(f'{country_names[id_US]} & ')
    lines.append(f'{results_multi[id_US_multi, 0, s]:.2f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 1, s]:.1f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 2, s]:.1f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 3, s]:.1f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 4, s]:.2f}\\% & ')
    lines.append(f'{results_multi[id_US_multi, 5, s]:.1f}\\% \\\\')

    lines.append('\\addlinespace[3pt]')
    avg_non_US = np.sum(E_non_US * results_multi[non_US_multi, :, s], axis=0) / np.sum(E_i_multi[non_US_multi])
    lines.append(f'non-US (average) & ')
    lines.append(f'{avg_non_US[0]:.2f}\\% & ')
    lines.append(f'{avg_non_US[1]:.1f}\\% & ')
    lines.append(f'{avg_non_US[2]:.1f}\\% & ')
    lines.append(f'{avg_non_US[3]:.1f}\\% & ')
    lines.append(f'{avg_non_US[4]:.2f}\\% & ')
    lines.append(f'{avg_non_US[5]:.1f}\\% \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    # Write to file
    output_path = os.path.join(output_dir, 'Table_4.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')

    print(f"\nTable 4 saved to: {output_path}")

    # Display summary
    print("\n" + "=" * 80)
    print("Table 4 Summary - USA Welfare Changes")
    print("=" * 80)
    print("\nPre-Retaliation:")
    print(f"  (1) USTR + one sector:        {results[id_US, 0, 0]:.2f}%")
    print(f"  (2) Optimal + one sector:     {results[id_US, 0, 3]:.2f}%")
    print(f"  (3) USTR + multi-sector:      {results_multi[id_US_multi, 0, 0]:.2f}%")
    print("\nPost-Retaliation:")
    print(f"  (1) Reciprocal + one sector:  {results[id_US, 0, 5]:.2f}%")
    print(f"  (2) Optimal + one sector:     {results[id_US, 0, 4]:.2f}%")
    print(f"  (3) Reciprocal + multi-sector:{results_multi[id_US_multi, 0, 1]:.2f}%")

    print("\n" + "=" * 80)
    print("Comparison with MATLAB Targets")
    print("=" * 80)
    print("\nUSA Welfare (Pre-retaliation, multi-sector):")
    print(f"  MATLAB target: 0.60%")
    print(f"  Python result: {results_multi[id_US_multi, 0, 0]:.2f}%")
    print("\nUSA Welfare (Post-retaliation, multi-sector):")
    print(f"  MATLAB target: -1.02%")
    print(f"  Python result: {results_multi[id_US_multi, 0, 1]:.2f}%")

    print("\n" + "=" * 80)
    print("Table 4 Generation Complete!")
    print("=" * 80)

if __name__ == '__main__':
    generate_table_4()
