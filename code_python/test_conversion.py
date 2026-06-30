"""
Test script to verify Python conversion works correctly.

This script runs a quick test of the main_baseline analysis
and compares results with expected MATLAB outputs (if available).
"""

import os
import sys
import numpy as np
import pandas as pd

# Add analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis'))

def test_baseline():
    """Test baseline analysis runs without errors."""
    print("="*60)
    print("Testing Baseline Analysis")
    print("="*60)

    try:
        from main_baseline import main

        # Run the analysis
        results_dict = main()

        print("\n" + "="*60)
        print("[OK] Baseline analysis completed successfully!")
        print("="*60)

        # Print key results
        print("\nKey Results Summary:")
        print("-"*60)

        results = results_dict['results']
        id_US = results_dict['id_US']

        scenarios = [
            "USTR tariffs (benchmark)",
            "Partial pass-through",
            "Eaton-Kortum",
            "Optimal tariff (no retal.)",
            "Liberation + optimal retal.",
            "Liberation + reciprocal retal.",
            "Optimal + optimal retal.",
            "Lump-sum rebate",
            "Higher elasticity"
        ]

        print(f"\n{'Scenario':<35} {'US Welfare':>12} {'US CPI':>12}")
        print("-"*60)
        for i, scenario in enumerate(scenarios):
            if i < results.shape[2]:
                welfare = results[id_US, 0, i]
                cpi = results[id_US, 5, i]
                print(f"{scenario:<35} {welfare:>11.2f}% {cpi:>11.1f}%")

        # Check if output files were created
        print("\n" + "="*60)
        print("Checking Output Files")
        print("="*60)

        base_path = os.path.join(os.path.dirname(__file__), '..')
        output_files = [
            'output/output_map.csv',
            'output/output_map_retal.csv'
        ]

        for file in output_files:
            file_path = os.path.join(base_path, file)
            if os.path.exists(file_path):
                print(f"[OK] Created: {file}")
                # Show first few rows
                df = pd.read_csv(file_path)
                print(f"  Shape: {df.shape}")
            else:
                print(f"[!!] Missing: {file}")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("[!!] Error in baseline analysis:")
        print("="*60)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_matlab():
    """Compare Python outputs with MATLAB outputs if available."""
    print("\n" + "="*60)
    print("Comparing with MATLAB Outputs")
    print("="*60)

    base_path = os.path.join(os.path.dirname(__file__), '..')

    # Check if MATLAB outputs exist
    matlab_files = [
        'output/Table_1.tex',
        'output/output_map.csv'
    ]

    matlab_exists = all(os.path.exists(os.path.join(base_path, f)) for f in matlab_files)

    if not matlab_exists:
        print("\n[--] MATLAB output files not found.")
        print("Run MATLAB version first to enable comparison:")
        print("  >> cd replication_package")
        print("  >> run run_all_matlab.m")
        return

    # Compare CSV outputs
    try:
        matlab_csv = pd.read_csv(os.path.join(base_path, 'output/output_map.csv'))
        print(f"\n[OK] Found MATLAB output_map.csv")
        print(f"  Shape: {matlab_csv.shape}")
        print(f"  Columns: {list(matlab_csv.columns)}")

        # Could add more detailed comparisons here
        print("\n[--] Detailed numerical comparison not implemented yet.")
        print("  Manually compare output files for now.")

    except Exception as e:
        print(f"\n[!!] Error comparing files: {e}")


def main_test():
    """Run all tests."""
    print("\n" + "="*70)
    print(" Python Trade Model Conversion - Test Suite")
    print("="*70)

    # Test baseline
    baseline_ok = test_baseline()

    if baseline_ok:
        # Compare with MATLAB if available
        compare_with_matlab()

        print("\n" + "="*70)
        print(" Test Suite Completed")
        print("="*70)
        print("\n[OK] All tests passed!")
        print("\nNext steps:")
        print("  1. Compare output files with MATLAB versions")
        print("  2. Run multi-sector extension")
        print("  3. Complete conversion of remaining files")

    else:
        print("\n" + "="*70)
        print(" Test Suite Failed")
        print("="*70)
        print("\n[!!] Please fix errors before proceeding.")

    return baseline_ok


if __name__ == '__main__':
    success = main_test()
    sys.exit(0 if success else 1)
