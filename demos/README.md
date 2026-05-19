# Demos

Run MIOFlow on the uploaded trifurcation data from the repo root:

```bash
conda run -n mioflow python demos/run_real_data_mioflow.py --dataset trifurcation --epochs 300
```

Run MIOFlow on the uploaded EB `adata_time.h5ad` data:

```bash
conda run -n mioflow python demos/run_real_data_mioflow.py --dataset adata_time --epochs 300 --max-points-per-time 1000
```

The real-data demo follows the notebook preprocessing: `trifurcation.csv` uses standardized `d1`/`d2` grouped by `samples`, and `adata_time.h5ad` uses standardized `obsm['X_phate']` grouped by `obs['time_label']`.

Outputs are written to `demos/outputs/<dataset>/` by default:

- `training_data.csv`
- `metadata.json`
- `loss_history.json`
- `loss_history.csv`
- `loss_history.png`
- `trajectories.npy`
- `trajectories.png`
- `mioflow_model.pt`

Use `--out-dir path/to/output` to write results somewhere else.

For a dependency-free toy smoke test, run:

```bash
conda run -n mioflow python demos/run_synthetic_mioflow.py --epochs 2 --points-per-time 12
```
