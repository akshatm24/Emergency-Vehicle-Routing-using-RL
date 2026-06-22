# Tools

This folder contains optional helper scripts used during development and validation.

- `run_notebook_audit.py`: executes a notebook and records per-cell timing, errors, and warning patterns.
- `probe_ablation_gradients.py`: diagnostic script for checking ablation tensor shapes and gradient flow.

These scripts are not required for reviewing the final project. Running them may execute notebook code, so avoid using them unless you intentionally want to reproduce the audits.
