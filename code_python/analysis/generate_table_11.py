"""
Generate Table 11: Multi-sector and IO model comparison
Combines results from baseline, IO, multi-sector baseline, and multi-sector IO models
"""

import numpy as np
import pandas as pd
import subprocess
import sys
import os

def run_multisector_models():
    """Run both multi-sector models to generate required data"""
    print("="* 80)
    print("Generating Table 11: Multi-sector Model Comparison")
    print("=" * 80)
    
    # Run multi-sector baseline model
    print("\nStep 1/2: Running multi-sector baseline model...")
    print("-" * 80)
    result = subprocess.run(
        ['python3', 'sub_multisector_baseline.py'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERROR: Multi-sector baseline failed!")
        print(result.stderr)
        return False
    print(result.stdout)
    
    # Run multi-sector IO model
    print("\nStep 2/2: Running multi-sector IO model...")
    print("-" * 80)
    result = subprocess.run(
        ['python3', 'sub_multisector_io.py'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"ERROR: Multi-sector IO failed!")
        print(result.stderr)
        return False
    print(result.stdout)
    
    return True

def generate_table_11():
    """Generate Table 11 LaTeX output"""
    print("\n" + "=" * 80)
    print("Generating Table 11 LaTeX file...")
    print("=" * 80)
    
    # Load baseline results
    print("\nLoading baseline model results...")
    baseline_data = np.load('../../python_output/baseline_results.npz')
    d_trade = baseline_data['d_trade']  # indices: 0=USTR, 5=USTR+retal, 7=multi_USTR, 8=multi_retal
    d_employment = baseline_data['d_employment']
    
    # Load IO results  
    print("Loading IO model results...")
    io_data = np.load('../../python_output/io_results.npz')
    d_trade_IO = io_data['d_trade_IO']  # indices: 0=USTR, 1=USTR+retal
    d_employment_IO = io_data['d_employment_IO']
    
    # Load multi-sector baseline results
    print("Loading multi-sector baseline results...")
    multi_base_data = np.load('../../python_output/multisector_baseline_results.npz')
    d_trade_multi = multi_base_data['d_trade_multi']  # indices: 0=USTR, 1=retal
    d_employment_multi = multi_base_data['d_employment_multi']
    
    # Load multi-sector IO results
    print("Loading multi-sector IO results...")
    multi_io_data = np.load('../../python_output/multisector_io_results.npz')
    d_trade_IO_multi = multi_io_data['d_trade_IO_multi']  # indices: 0=USTR, 1=retal
    d_employment_IO_multi = multi_io_data['d_employment_IO_multi']
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    
    table_lines = []
    table_lines.append('\\begin{tabular}{lcccccccccc}')
    table_lines.append('\\toprule')
    table_lines.append('& \\multicolumn{4}{c}{before retaliation} && \\multicolumn{4}{c}{after retaliation} \\\\')
    table_lines.append('\\cmidrule(lr){2-5} \\cmidrule(lr){7-10}')
    table_lines.append('main & IO & multi & multi + IO  && main & IO & multi & multi + IO \\\\')
    table_lines.append('\\midrule')
    
    # Global trade-to-GDP row
    table_lines.append('$\\Delta$ global trade-to-GDP &')
    table_lines.append(f'{d_trade[0]:.1f}\\% & ')  # main before
    table_lines.append(f'{d_trade_IO[0]:.1f}\\% &')  # IO before
    table_lines.append(f'{d_trade_multi[0]:.1f}\\% & ')  # multi before
    table_lines.append(f'{d_trade_IO_multi[0]:.1f}\\% && ')  # multi+IO before
    table_lines.append(f'{d_trade[5]:.1f}\\% & ')  # main after (index 5 = USTR+retal)
    table_lines.append(f'{d_trade_IO[1]:.1f}\\% &')  # IO after
    table_lines.append(f'{d_trade_multi[1]:.1f}\\% & ')  # multi after
    table_lines.append(f'{d_trade_IO_multi[1]:.1f}\\% \\\\ ')  # multi+IO after
    
    table_lines.append(' \\addlinespace[3pt]')
    
    # Global employment row
    table_lines.append('$\\Delta$ global employment &')
    table_lines.append(f'{d_employment[0]:.2f}\\% & ')  # main before
    table_lines.append(f'{d_employment_IO[0]:.2f}\\% &')  # IO before
    table_lines.append(f'{d_employment_multi[0]:.2f}\\% & ')  # multi before
    table_lines.append(f'{d_employment_IO_multi[0]:.2f}\\% && ')  # multi+IO before
    table_lines.append(f'{d_employment[5]:.2f}\\% & ')  # main after
    table_lines.append(f'{d_employment_IO[1]:.2f}\\% &')  # IO after
    table_lines.append(f'{d_employment_multi[1]:.2f}\\% & ')  # multi after
    table_lines.append(f'{d_employment_IO_multi[1]:.2f}\\% \\\\ ')  # multi+IO after
    
    table_lines.append(' \\addlinespace[3pt]')
    table_lines.append(' \\bottomrule')
    table_lines.append('\\end{tabular}')
    
    # Write to file
    output_path = '../../python_output/Table_11.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(table_lines))
        f.write('\n')
    
    print(f"\nTable 11 saved to: {output_path}")
    
    # Display summary
    print("\n" + "=" * 80)
    print("Table 11 Summary")
    print("=" * 80)
    print("\nBefore Retaliation:")
    print(f"  main:       Δ trade = {d_trade[0]:6.1f}%  |  Δ employment = {d_employment[0]:6.2f}%")
    print(f"  IO:         Δ trade = {d_trade_IO[0]:6.1f}%  |  Δ employment = {d_employment_IO[0]:6.2f}%")
    print(f"  multi:      Δ trade = {d_trade_multi[0]:6.1f}%  |  Δ employment = {d_employment_multi[0]:6.2f}%")
    print(f"  multi + IO: Δ trade = {d_trade_IO_multi[0]:6.1f}%  |  Δ employment = {d_employment_IO_multi[0]:6.2f}%")
    
    print("\nAfter Retaliation:")
    print(f"  main:       Δ trade = {d_trade[5]:6.1f}%  |  Δ employment = {d_employment[5]:6.2f}%")
    print(f"  IO:         Δ trade = {d_trade_IO[1]:6.1f}%  |  Δ employment = {d_employment_IO[1]:6.2f}%")
    print(f"  multi:      Δ trade = {d_trade_multi[1]:6.1f}%  |  Δ employment = {d_employment_multi[1]:6.2f}%")
    print(f"  multi + IO: Δ trade = {d_trade_IO_multi[1]:6.1f}%  |  Δ employment = {d_employment_IO_multi[1]:6.2f}%")
    
    # Load MATLAB target for comparison
    print("\n" + "=" * 80)
    print("Comparison with MATLAB Target")
    print("=" * 80)
    matlab_values = {
        'before': {
            'main_trade': -9.4, 'io_trade': -10.8, 'multi_trade': -5.5, 'multi_io_trade': -4.1,
            'main_emp': -0.02, 'io_emp': -0.12, 'multi_emp': -0.05, 'multi_io_emp': -0.08
        },
        'after': {
            'main_trade': -11.6, 'io_trade': -12.4, 'multi_trade': -6.9, 'multi_io_trade': -4.9,
            'main_emp': -0.15, 'io_emp': -0.58, 'multi_emp': -0.05, 'multi_io_emp': -0.26
        }
    }
    
    print("\nBefore Retaliation (MATLAB → Python):")
    print(f"  main trade:       {matlab_values['before']['main_trade']:6.1f}% → {d_trade[0]:6.1f}%")
    print(f"  IO trade:         {matlab_values['before']['io_trade']:6.1f}% → {d_trade_IO[0]:6.1f}%")
    print(f"  multi trade:      {matlab_values['before']['multi_trade']:6.1f}% → {d_trade_multi[0]:6.1f}%")
    print(f"  multi+IO trade:   {matlab_values['before']['multi_io_trade']:6.1f}% → {d_trade_IO_multi[0]:6.1f}%")
    
    print("\nAfter Retaliation (MATLAB → Python):")
    print(f"  main trade:       {matlab_values['after']['main_trade']:6.1f}% → {d_trade[5]:6.1f}%")
    print(f"  IO trade:         {matlab_values['after']['io_trade']:6.1f}% → {d_trade_IO[1]:6.1f}%")
    print(f"  multi trade:      {matlab_values['after']['multi_trade']:6.1f}% → {d_trade_multi[1]:6.1f}%")
    print(f"  multi+IO trade:   {matlab_values['after']['multi_io_trade']:6.1f}% → {d_trade_IO_multi[1]:6.1f}%")
    
    print("\n" + "=" * 80)
    print("Table 11 Generation Complete!")
    print("=" * 80)

def main():
    # Run multi-sector models
    if not run_multisector_models():
        print("\nERROR: Multi-sector model execution failed!")
        sys.exit(1)
    
    # Generate Table 11
    generate_table_11()

if __name__ == '__main__':
    main()
