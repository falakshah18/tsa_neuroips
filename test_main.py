#!/usr/bin/env python
# test_main.py
"""
Test runner for the TSA NeuroIPS project.

Usage:
    pytest test_main.py -v
    python test_main.py
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    # Run pytest
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        capture_output=False,
    )

    return result.returncode


def run_import_tests():
    """Test that all imports work."""
    print("\n" + "=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)

    try:
        import torch
        import spikingjelly
        import tonic
        import numpy as np
        import matplotlib
        import seaborn
        import yaml
        try:
            import wandb
        except ImportError:
            wandb = None
        import scipy
        import sklearn

        print("✅ All imports successful")
        print(f"   torch: {torch.__version__}")
        print(f"   spikingjelly: {spikingjelly.__version__ if hasattr(spikingjelly, '__version__') else 'installed'}")
        print(f"   numpy: {np.__version__}")

        return 0
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1

def main():
    """Run all tests."""
    import_failed = run_import_tests()

    if import_failed:
        print("\n❌ Import tests failed. Check requirements.")
        return 1

    test_failed = run_tests()

    if test_failed:
        print("\n❌ Unit tests failed.")
        return 1

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())