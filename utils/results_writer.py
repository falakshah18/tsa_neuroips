"""
Centralised results saving — JSON for full data, CSV for quick analysis.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_results_json(
    results: Dict[str, Any],
    save_dir: str = "./results",
    filename: str = "results.json",
    *,
    indent: int = 2,
) -> str:
    """Save a dictionary of results to a JSON file. Returns the full path."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / filename
    with open(full_path, "w") as f:
        json.dump(results, f, indent=indent, sort_keys=True)
    return str(full_path)


def save_results_csv(
    rows: List[Dict[str, Any]],
    save_dir: str = "./results",
    filename: str = "results.csv",
) -> str:
    """Save a list of dicts (one per row) to a CSV file. Returns the full path."""
    if not rows:
        return ""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / filename
    fieldnames = list(rows[0].keys())
    with open(full_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(full_path)


def flatten_results(
    all_results: Dict[str, Dict[str, List[Dict]]],
) -> List[Dict[str, Any]]:
    """
    Flatten the nested results dict from BaselineComparison/TSA experiments
    into a list of flat dicts suitable for CSV export.

    Input structure:
        {dataset: {algorithm: [{seed, test_acc, test_energy, ...}, ...], ...}, ...}

    Output:
        [{dataset, algorithm, seed, test_acc, test_energy, ...}, ...]
    """
    rows: List[Dict[str, Any]] = []
    for dataset, algos in all_results.items():
        for algo, seed_results in algos.items():
            for r in seed_results:
                row = {"dataset": dataset, "algorithm": algo}
                row.update(r)
                rows.append(row)
    return rows
