"""
run_sector_analyses.py
======================
Master orchestrator for Phase 2 sector-specific analyses.

Runs all three sector modules in sequence:
  1. Manufacturing / Steel & Aluminum
  2. Pharmaceuticals
  3. Retail / Consumer Goods (distributional incidence)

Each module reads pre-computed GE results (baseline_results.npz and
multisector_io_results.npz) and produces LaTeX tables, figures,
and a compressed-results .npz file in python_output/.

Usage
-----
    cd code_python/analysis
    python run_sector_analyses.py

Prerequisite files (in python_output/):
    baseline_results.npz           -- single-sector GE results (194 countries, 9 scenarios)
    multisector_io_results.npz     -- multi-sector IO-adjusted GE results (181 countries, 2 scenarios)

If either prerequisite is missing the script will print instructions for
generating them and then exit.
"""

import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Path setup – allow running from either the repo root or code_python/analysis
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "python_output")

# Make sure sector modules are importable regardless of working directory
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------
REQUIRED_FILES = {
    "baseline_results.npz": (
        "Generate with:  python code_python/analysis/main_baseline.py"
    ),
    "multisector_io_results.npz": (
        "Generate with:  python code_python/analysis/sub_multisector_io.py"
    ),
}


def check_prerequisites():
    missing = []
    for fname, hint in REQUIRED_FILES.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            missing.append((fname, hint))
    if missing:
        print("ERROR: Required GE result files not found in python_output/\n")
        for fname, hint in missing:
            print(f"  Missing: {fname}")
            print(f"  {hint}\n")
        print("Run the GE models first, then re-run this script.")
        sys.exit(1)
    print("Prerequisites check passed.\n")


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def run_module(label, module_name, func_name):
    """Import *module_name* and call *func_name*(), reporting timing."""
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        result = func()
        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s\n")
        return True, result
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  FAILED after {elapsed:.1f}s")
        print(f"  Error: {exc}")
        traceback.print_exc()
        return False, None


def collect_outputs(prefix):
    """Return a sorted list of output files whose names start with *prefix*."""
    files = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.startswith(prefix):
            files.append(fname)
    return files


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(outcomes):
    """Print a final summary table of what was produced."""
    print("\n" + "=" * 70)
    print("  SECTOR ANALYSIS SUMMARY")
    print("=" * 70)

    all_ok = all(ok for _, ok, _ in outcomes)

    for label, ok, prefix in outcomes:
        status = "OK" if ok else "FAILED"
        badge = "[OK]" if ok else "[!!]"
        print(f"\n{badge} {label:45s}  {status}")
        if ok:
            files = collect_outputs(prefix)
            if files:
                for f in files:
                    ext = os.path.splitext(f)[1]
                    kind = {".tex": "LaTeX", ".png": "Figure", ".npz": "Data"}.get(ext, ext)
                    print(f"     {kind:8s}  python_output/{f}")
            else:
                print("     (no output files found)")

    print("\n" + "-" * 70)
    if all_ok:
        print("All sector analyses completed successfully.")
        print(f"Output directory: {OUTPUT_DIR}")
    else:
        failed = [lbl for lbl, ok, _ in outcomes if not ok]
        print(f"WARNING: {len(failed)} module(s) failed: {', '.join(failed)}")
        print("Check the error messages above for details.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  Liberation Day Tariff Analysis -- Phase 2 Sector Analyses")
    print("=" * 70 + "\n")

    check_prerequisites()

    outcomes = []

    # --- 1. Manufacturing / Steel & Aluminum ---
    ok, _ = run_module(
        "1/3  Manufacturing & Steel/Aluminum Analysis",
        "sector_manufacturing",
        "analyze_manufacturing",
    )
    outcomes.append(("Manufacturing / Steel & Aluminum", ok, "sector_manufacturing"))

    # --- 2. Pharmaceuticals ---
    ok, _ = run_module(
        "2/3  Pharmaceuticals Analysis",
        "sector_pharma",
        "analyze_pharma",
    )
    outcomes.append(("Pharmaceuticals", ok, "sector_pharma"))

    # --- 3. Retail / Consumer Goods ---
    ok, _ = run_module(
        "3/3  Retail / Consumer Goods (Distributional Incidence) Analysis",
        "sector_retail",
        "analyze_retail",
    )
    outcomes.append(("Retail / Consumer Goods", ok, "sector_retail"))

    print_summary(outcomes)

    # Exit with non-zero code if any module failed (useful for CI)
    if not all(ok for _, ok, _ in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    main()
