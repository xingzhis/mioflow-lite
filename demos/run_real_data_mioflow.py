"""Run MIOFlow on the real demo datasets in data/.

Examples:
    python demos/run_real_data_mioflow.py --dataset trifurcation --epochs 300
    python demos/run_real_data_mioflow.py --dataset adata_time --epochs 300 --max-points-per-time 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mioflow import ODEFunc, TimeSeriesDataset, infer, train_mioflow  # noqa: E402


TIME_LABEL_TO_BIN = {
    "Day 00-03": 0.0,
    "Day 06-09": 1.0,
    "Day 12-15": 2.0,
    "Day 18-21": 3.0,
    "Day 24-27": 4.0,
}


def load_trifurcation(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "trifurcation.csv")
    features = StandardScaler().fit_transform(df[["d1", "d2"]].to_numpy())
    return pd.DataFrame({"d1": features[:, 0], "d2": features[:, 1], "samples": df["samples"].astype(float)})


def load_adata_time(data_dir: Path) -> pd.DataFrame:
    adata = sc.read_h5ad(data_dir / "adata_time.h5ad")
    if "X_phate" not in adata.obsm:
        raise ValueError("adata_time.h5ad must contain adata.obsm['X_phate']")
    if "time_label" not in adata.obs:
        raise ValueError("adata_time.h5ad must contain adata.obs['time_label']")

    features = StandardScaler().fit_transform(np.asarray(adata.obsm["X_phate"]))
    samples = adata.obs["time_label"].map(TIME_LABEL_TO_BIN)
    if samples.isna().any():
        missing = sorted(set(adata.obs.loc[samples.isna(), "time_label"]))
        raise ValueError(f"Unmapped time labels in adata_time.h5ad: {missing}")

    return pd.DataFrame({"d1": features[:, 0], "d2": features[:, 1], "samples": samples.astype(float).to_numpy()})


def subsample_by_time(df: pd.DataFrame, max_points_per_time: int | None, seed: int) -> pd.DataFrame:
    if max_points_per_time is None:
        return df

    parts = []
    for _, group in df.groupby("samples", sort=True):
        n = min(max_points_per_time, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def make_dataset(df: pd.DataFrame) -> TimeSeriesDataset:
    time_series_data = []
    for time_value in np.unique(df["samples"]):
        points = df.loc[df["samples"] == time_value, ["d1", "d2"]].to_numpy(dtype=np.float32)
        time_series_data.append((points, float(time_value)))
    return TimeSeriesDataset(time_series_data)


def save_history(history: dict, out_dir: Path) -> None:
    history_json = {key: [float(value) for value in values] for key, values in history.items()}
    (out_dir / "loss_history.json").write_text(json.dumps(history_json, indent=2) + "\n")

    with (out_dir / "loss_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        keys = list(history.keys())
        writer.writerow(keys)
        for row in zip(*(history[key] for key in keys)):
            writer.writerow(row)


def save_loss_plot(history: dict, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["total_loss"], label="total", linewidth=2.5)
    plt.plot(history["epoch"], history["ot_loss"], label="ot")
    plt.plot(history["epoch"], history["energy_loss"], label="energy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_history.png", dpi=180)
    plt.close()


def save_trajectory_plot(df: pd.DataFrame, trajectories: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(df["d1"], df["d2"], c=df["samples"], cmap="viridis", s=10, alpha=0.35)
    for trajectory in trajectories:
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="black", linewidth=0.9, alpha=0.35)
    plt.colorbar(scatter, label="time")
    plt.xlabel("d1")
    plt.ylabel("d2")
    plt.tight_layout()
    plt.savefig(out_dir / "trajectories.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIOFlow on data/trifurcation.csv or data/adata_time.h5ad.")
    parser.add_argument("--dataset", choices=["trifurcation", "adata_time"], default="trifurcation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--lambda-ot", type=float, default=1.0)
    parser.add_argument("--lambda-energy", type=float, default=None)
    parser.add_argument("--energy-time-steps", type=int, default=20)
    parser.add_argument("--scheduler-min-lr", type=float, default=5e-4)
    parser.add_argument("--max-points-per-time", type=int, default=None)
    parser.add_argument("--trajectory-points", type=int, default=250)
    parser.add_argument("--trajectory-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir or Path("demos") / "outputs" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "trifurcation":
        df = load_trifurcation(args.data_dir)
        lambda_energy = 0.1 if args.lambda_energy is None else args.lambda_energy
    else:
        df = load_adata_time(args.data_dir)
        lambda_energy = 0.075 if args.lambda_energy is None else args.lambda_energy

    df = subsample_by_time(df, args.max_points_per_time, args.seed)
    dataset = make_dataset(df)
    model = ODEFunc(input_dim=2, hidden_dim=args.hidden_dim)

    history = train_mioflow(
        model=model,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        lambda_ot=args.lambda_ot,
        lambda_density=0.0,
        lambda_energy=lambda_energy,
        energy_time_steps=args.energy_time_steps,
        scheduler_type="cosine",
        scheduler_min_lr=args.scheduler_min_lr,
    )

    x0_full = dataset.get_initial_condition()
    trajectory_count = min(args.trajectory_points, x0_full.size(0))
    indices = torch.randperm(x0_full.size(0))[:trajectory_count]
    x0 = x0_full[indices].to(args.device)
    t_seq = torch.linspace(min(dataset.times), max(dataset.times), args.trajectory_steps, device=args.device)
    with torch.no_grad():
        trajectories = infer(x0=x0, model=model, t_seq=t_seq).permute(1, 0, 2).cpu().numpy()

    df.to_csv(out_dir / "training_data.csv", index=False)
    np.save(out_dir / "trajectories.npy", trajectories)
    torch.save(model.state_dict(), out_dir / "mioflow_model.pt")
    save_history(history, out_dir)
    save_loss_plot(history, out_dir)
    save_trajectory_plot(df, trajectories, out_dir)

    metadata = {
        "dataset": args.dataset,
        "num_points": int(len(df)),
        "time_counts": {str(k): int(v) for k, v in df["samples"].value_counts().sort_index().items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda_ot": args.lambda_ot,
        "lambda_energy": lambda_energy,
        "trajectory_shape": list(trajectories.shape),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Final losses:", {key: round(float(history[key][-1]), 6) for key in ["total_loss", "ot_loss", "energy_loss"]})
    print("Trajectory tensor shape:", tuple(trajectories.shape))
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
