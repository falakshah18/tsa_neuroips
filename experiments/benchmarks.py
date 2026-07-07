# experiments/benchmarks.py
"""
Benchmarking utilities for comparing TSA against baselines.

This module provides lightweight benchmark wrappers that delegate to the
full BaselineComparison framework in experiments/baseline_comparison.py.
"""

from experiments.baseline_comparison import BaselineComparison

__all__ = ['BaselineComparison']