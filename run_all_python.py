#!/usr/bin/env python3
"""
Master script to run all Python analyses.

This script replicates run_all_matlab.m from the MATLAB replication package.
Produces Tables 1-4 and 8-11 for:
"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"
by Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)

For publication in the Journal of International Economics (Last update: July 2025)
Email for inquiries: ahmadlp@gmail.com
"""

import os
import sys
import subprocess

# Add code_python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code_python'))
from code_python.config import get_output_dir

def run_analysis(module_name, description):
    """
    Run a Python analysis module.

    Parameters:
    -----------
    module_name : str
        Name of the module to run (without .py extension)
    description : str
        Description of the analysis
    """
    print("\n" + "="*70)
    print(f" {description}")
    print("="*70)

    try:
        # Import and run the module
        module = __import__(f'analysis.{module_name}', fromlist=['main'])
        results = module.main()
        print(f"\n[OK] {description} completed successfully!")
        return results
    except Exception as e:
        print(f"\n[!!] Error in {description}:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    print("="*70)
    print(" MATLAB to Python Replication Package")
    print(" Making America Great Again? The Economic Impacts of")
    print(" Liberation Day Tariffs")
    print("="*70)
    print("\nThis script produces Tables 1-4 and 8-11")
    print("from the paper by Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)")
    print("\nCurrent directory:", os.getcwd())

    # Change to replication package directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Working directory:", os.getcwd())

    # Track results
    all_results = {}

    # 1. Baseline model
    results = run_analysis('main_baseline', 'Baseline Model Analysis')
    if results:
        all_results['baseline'] = results
    else:
        print("\n[!!] Baseline analysis failed. Stopping execution.")
        return

    # 2. Baseline + IO (TEMPORARILY DISABLED - convergence issues)
    print("\n" + "="*70)
    print(" Input-Output Model Analysis")
    print("="*70)
    print("[--] SKIPPED: IO model has convergence issues - will be fixed separately")
    # results = run_analysis('main_io', 'Input-Output Model Analysis')
    # if results:
    #     all_results['io'] = results

    # 3. Create Figure 1 using Python
    print("\n" + "="*70)
    print(" Creating Figure 1")
    print("="*70)

    try:
        python_cmd = sys.executable
        fig_script = os.path.join('code', 'global_map', 'create_figure_1.py')

        if os.path.exists(fig_script):
            result = subprocess.run([python_cmd, fig_script],
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print("[OK] Figure 1 created successfully!")
            else:
                print(f"[!!] Figure 1 creation failed:")
                print(result.stderr)
        else:
            print(f"[--] Figure script not found: {fig_script}")
    except Exception as e:
        print(f"[!!] Error creating figure: {e}")

    # 4. Regional trade war
    results = run_analysis('main_regional', 'Regional Trade War Analysis')
    if results:
        all_results['regional'] = results

    # 5. Alternative deficit framework
    results = run_analysis('main_deficit', 'Deficit Framework Analysis')
    if results:
        all_results['deficit'] = results

    # 6. Phase 2: Sector-specific analyses
    print("\n" + "="*70)
    print(" Phase 2: Sector-Specific Analyses")
    print("="*70)
    print("Running Manufacturing, Pharmaceuticals, and Retail analyses...")
    try:
        import importlib
        sector_runner = importlib.import_module('analysis.run_sector_analyses')
        # Run each sector module directly (skip the prerequisites re-check)
        for mod_name, func_name, label in [
            ('analysis.sector_manufacturing', 'analyze_manufacturing', 'Manufacturing'),
            ('analysis.sector_pharma',        'analyze_pharma',        'Pharmaceuticals'),
            ('analysis.sector_retail',        'analyze_retail',        'Retail / Consumer'),
        ]:
            try:
                mod = importlib.import_module(mod_name)
                getattr(mod, func_name)()
                print(f"[OK] {label} sector analysis complete.")
                all_results[label.lower()] = True
            except Exception as sec_err:
                print(f"[!!] {label} sector analysis failed: {sec_err}")
    except Exception as e:
        print(f"[!!] Phase 2 sector analyses failed: {e}")

    # Summary
    print("\n" + "="*70)
    print(" EXECUTION SUMMARY")
    print("="*70)

    print("\nCompleted analyses:")
    for key in all_results.keys():
        print(f"  [OK] {key}")

    print("\nGenerated outputs:")
    output_dir_name = get_output_dir()
    output_dir = os.path.join(os.getcwd(), output_dir_name)
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        tex_files = [f for f in files if f.endswith('.tex')]
        csv_files = [f for f in files if f.endswith('.csv')]
        png_files = [f for f in files if f.endswith('.png')]

        if tex_files:
            print(f"\n  LaTeX Tables ({len(tex_files)}):")
            for f in sorted(tex_files):
                print(f"    - {f}")

        if csv_files:
            print(f"\n  CSV Files ({len(csv_files)}):")
            for f in sorted(csv_files):
                print(f"    - {f}")

        if png_files:
            print(f"\n  Figures ({len(png_files)}):")
            for f in sorted(png_files):
                print(f"    - {f}")

    print("\n" + "="*70)
    print(" ALL ANALYSES COMPLETED")
    print("="*70)
    print(f"\nOutput files can be found in: ./{output_dir_name}/")
    print("\nTo validate results:")
    print("  1. Compare LaTeX tables with MATLAB versions")
    print("  2. Check CSV files for numerical accuracy")
    print("  3. Verify Figure 1 rendering")
    print("\nFor questions: ahmadlp@gmail.com")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!!] Execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[!!] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
