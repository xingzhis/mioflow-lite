# Demos

Run a small self-contained MIOFlow example from the repo root:

```bash
conda run -n mioflow python demos/run_synthetic_mioflow.py --epochs 2 --points-per-time 12
```

The demo generates synthetic 2D time-series data, trains a small ODE model, and writes outputs to `demos/outputs/synthetic_mioflow/` by default:

- `loss_history.json`
- `loss_history.csv`
- `loss_history.png`
- `trajectories.npy`
- `trajectories.png`

Use `--out-dir path/to/output` to write results somewhere else.
