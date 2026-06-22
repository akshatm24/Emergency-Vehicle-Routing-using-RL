# Reproducibility Notes

The notebook has already been executed and includes outputs. Do not re-run the full notebook unless compute is available.

## Environment

The final notebook metadata reports:

- Python: 3.10.0
- Kernel: Python 3

Install dependencies with:

```bash
pip install -r requirements.txt
```

PyTorch Geometric is optional. If it is unavailable, the notebook contains fallback logic for the encoder.

## Compute Warning

Several cells are expensive:

- OSM graph download and projection
- Node2Vec training on the Kanpur road graph
- DRL training and extended training
- Ablation training
- Diagnostic learning-rate sweeps

For review, use the already-run notebook outputs and artifacts. For a fresh run, a GPU runtime such as Google Colab is recommended.

## Re-running Safely

If re-running is required:

1. Run Phase 1 first to build the road graph and hospital data.
2. Keep `CANDIDATE_POOL_SIZE`, `NUM_INCIDENTS`, and random seeds fixed for comparable results.
3. Use cached Node2Vec embeddings when available.
4. Avoid increasing incident count without replacing brute-force optimal evaluation, because exact permutation search grows factorially.
5. Save outputs after long training runs so results are not lost.

## External Data Stability

The notebook downloads current OpenStreetMap data. Since OSM is edited over time, exact node IDs, hospital counts, and route costs may change in future runs. The committed notebook and artifacts preserve the completed run used for submission.

## Audit Artifact

`artifacts/final_run/notebook_audit.json` records a prior successful notebook execution with no errors. The final v7 notebook also includes already-run outputs from the updated CUDA/PyTorch run.
