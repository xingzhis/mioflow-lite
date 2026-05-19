# Demos

Run a small self-contained MIOFlow example from the repo root:

```bash
conda run -n mioflow python demos/run_synthetic_mioflow.py --epochs 2 --points-per-time 12
```

The demo generates synthetic 2D time-series data, trains a small ODE model, and prints the final losses plus a sample inferred trajectory shape.
