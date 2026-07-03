# Changelog

All notable changes to this project are documented in this file, grouped by date. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

This project doesn't use version numbers yet (no `v1.0.0` release has been made — see README Status section for why).

## 2026-07-02

### Fixed
- `--algorithm`/`--dataset` CLI flags were silently ignored during `--quick` runs; `BaselineComparison` never accepted an `algorithms` filter at all, so it always ran all 6 algorithms regardless of what was requested. Quick mode also unconditionally forced the dataset to `nmnist` even when a different dataset was explicitly passed (in two separate places).
- `RuntimeError: expected scalar type Short but found Float` on all real N-MNIST/SHD data — tonic's `ToFrame` yields `int16` event-count frames, which were never cast to float before reaching a Conv2d/Linear layer.
- SHD dataset loading used the wrong `sensor_size` (assumed a polarity channel that doesn't exist in tonic's real SHD data), and the resulting frames weren't flattened to the shape the FC-based baseline models expect.
- ANN-to-SNN baseline: three stacked bugs on the training path — missing `ConvertedSNN` import, missing time-averaging before the plain ANN's Conv2d layers, and a no-op transpose in the post-conversion evaluation loop that also miscounted batch size.
- ANN-to-SNN baseline on SHD specifically: was hardcoded to attempt a Conv2d-based spiking conversion, even though there's no valid FC-to-spiking-Conv2d conversion path; now evaluates the trained ANN directly instead.
- Infinite recursion (`RecursionError`) in `reset()` across 4 baseline files (surrogate gradient, STDP, e-prop, ANN-to-SNN) — each model's `reset()` called `functional.reset_net(self)`, which is self-referential.
- `training/trainer_v2.py` was a broken placeholder file (literally the text `[full corrected code here]`), causing a `SyntaxError` on import.
- `submission/rebuttal_preparation.py` had an unterminated triple-quoted string, also a `SyntaxError` on import.
- `models/tst_v2.py`'s `get_energy_breakdown()` was missing the `total_energy_J` key asserted by tests.
- Bio-plausibility scoring double-counted a precomputed `total_score` field already present in config YAMLs as an extra truthy criterion, inflating totals (e.g. STDP showed 5/5 instead of 4/5); the console report also displayed all-❌ regardless of actual values due to a dict-shape mismatch.
- `experiments/ablations.py`'s neuron-parameter ablation crashed with `TypeError` — `TemporalSpikingTransformer` never accepted the `learnable_tau`/`learnable_threshold` kwargs it was called with.
- Reverted a set of commits (since removed from history) that replaced `experiments/baseline_comparison.py`'s real logic with a non-functional stub while claiming "NeurIPS 2027 camera-ready" completeness — `run()` did nothing and `generate_report()` returned a hardcoded fake success message regardless of what actually ran.

### Added
- Live training progress bar (`tqdm`) with running loss, so slow CPU epochs on real data don't look identical to a hang.
- Algorithm name normalization layer so both display-style (`TTFS`) and snake_case (`ttfs`) identifiers resolve correctly.
- Comprehensive docstrings for `models/tst_v2.py` (including the TSA membrane-dynamics math) and `training/trainer_v2.py`.
- Accurate README and `docs/setup.md`, including a documented limitation (TSA's Conv2d patch embedding doesn't support SHD's 1D data — N-MNIST only for TSA) and hardware requirements informed by an actual CPU out-of-memory failure encountered during testing.
- `pyyaml`/`scipy` made explicit in `requirements.txt`.

### Changed
- Repo cleanup: removed an accidentally-committed embedded git repository and stray tensorboard log files from the working tree.

## 2026-07-01

### Added
- Initial commit: full TSA NeurIPS project scaffold — TemporalSpikingTransformer/LearnableTSA architecture, five baseline SNN implementations (surrogate gradient, ANN-to-SNN, supervised STDP, e-prop, TTFS), training pipeline, experiment scripts (baseline comparison, ablations, statistical validation), theory proofs, hardware energy estimation, and LaTeX paper source.
