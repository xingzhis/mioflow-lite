"""Minimal synthetic MIOFlow demo.

Run from the repo root:
    python demos/run_synthetic_mioflow.py --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mioflow import ODEFunc, TimeSeriesDataset, infer, train_mioflow  # noqa: E402


def make_synthetic_series(points_per_time: int, seed: int) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    data = []

    for t in range(3):
        center = np.array([0.6 * t, 0.2 * np.sin(t)], dtype=np.float32)
        points = center + rng.normal(scale=0.08, size=(points_per_time, 2)).astype(np.float32)
        data.append((points, float(t)))

    return TimeSeriesDataset(data)


def save_history(history: dict, out_dir: Path) -> None:
    history_json = {key: [float(value) for value in values] for key, values in history.items()}
    (out_dir / "loss_history.json").write_text(json.dumps(history_json, indent=2) + "\n")

    with (out_dir / "loss_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        keys = list(history.keys())
        writer.writerow(keys)
        for row in zip(*(history[key] for key in keys)):
            writer.writerow(row)


def save_plots(dataset: TimeSeriesDataset, history: dict, trajectories: torch.Tensor, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["total_loss"], label="total")
    plt.plot(history["epoch"], history["ot_loss"], label="ot")
    plt.plot(history["epoch"], history["energy_loss"], label="energy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    for points, time_value in dataset.time_series_data:
        points_np = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
        plt.scatter(points_np[:, 0], points_np[:, 1], s=18, alpha=0.65, label=f"t={time_value:g}")

    trajectories_np = trajectories.numpy()
    for trajectory in trajectories_np:
        plt.plot(trajectory[:, 0], trajectory[:, 1], color="black", linewidth=1.2, alpha=0.8)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color="black", s=18)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny synthetic MIOFlow demo.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--points-per-time", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("demos/outputs/synthetic_mioflow"))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_synthetic_series(args.points_per_time, args.seed)
    model = ODEFunc(input_dim=2, hidden_dim=args.hidden_dim)

    history = train_mioflow(
        model=model,
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=None,
        learning_rate=1e-3,
        device=args.device,
        lambda_ot=1.0,
        lambda_ot_per_interval=[0.5, 0.25],
        lambda_density=0.0,
        lambda_energy=0.1,
        lambda_energy_per_interval=[0.05, 0.01],
        energy_time_steps=3,
    )

    x0 = torch.as_tensor(dataset.get_initial_condition()[:5], dtype=torch.float32, device=args.device)
    t_seq = torch.linspace(0.0, 2.0, 5, device=args.device)
    with torch.no_grad():
        trajectories = infer(x0, model, t_seq).cpu()

    np.save(args.out_dir / "trajectories.npy", trajectories.numpy())
    save_history(history, args.out_dir)
    save_plots(dataset, history, trajectories, args.out_dir)

    final_losses = {
        "total": history["total_loss"][-1],
        "ot": history["ot_loss"][-1],
        "energy": history["energy_loss"][-1],
    }
    print("Final losses:", {key: round(value, 6) for key, value in final_losses.items()})
    print("Trajectory tensor shape:", tuple(trajectories.shape))
    print(f"Saved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
