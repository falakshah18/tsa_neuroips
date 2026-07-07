# Contributing

This is an active research codebase for a NeurIPS submission. While external
contributions aren't the primary focus, bug reports and fixes are welcome.

## Before You Start

- Check [open issues](https://github.com/falakshah18/tsa_neuroips/issues) for
  existing bug reports or feature requests.
- Open a new issue first if you want to discuss a change.

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Ensure 18 tests still pass:
   ```bash
   python -m pytest tests/ -q
   ```
3. Run linting if you have `ruff` installed:
   ```bash
   ruff check . # optional
   ```
4. Open a PR with a clear description of the problem and the fix.

## Code Conventions

- Python 3.10+ with type hints on all public functions.
- SpikingJelly `activation_based` multi-step API: `[T, B, ...]` tensors.
- DataLoader yields `[B, T, ...]`; `_prepare_batch` transposes to `[T, B, ...]`.
- No fabricated benchmark numbers in the paper LaTeX — use `[RESULTS PENDING]`.
