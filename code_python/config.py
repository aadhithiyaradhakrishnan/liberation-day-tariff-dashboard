"""
Configuration file for output directory.

This allows switching between 'output' (Matlab outputs) and 'python_output' (Python outputs)
to keep them separate.
"""

import os

# Set output directory - change this to 'output' to use original directory
OUTPUT_DIR = os.environ.get('PYTHON_OUTPUT_DIR', 'python_output')

def get_output_dir():
    """Get the output directory for Python runs."""
    return OUTPUT_DIR
