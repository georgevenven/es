"""Grid search runner for ES configs with reward curves and validation accuracy."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import torch

from data_loader import get_mnist_dataloaders
from model import ESConfig, EvolutionStrategyCNN
from train import evaluate_accuracy

IMG_DIR = Path(__file__).resolve().parent.parent / "imgs"
# Fixed settings (matching the provided train.py command).
TIME_STEPS = 16
BAG_SIZE = 256
SIGMA = 0.05
ALPHA = 3e-3
NPOP = 500 * 4  # "vmap by a factor of 4"
EPOCHS = 50
STEPS_PER_EPOCH = 500
# Sweep recurrent sparsity from 0% to 90% in 10% increments.
RECURRENT_SPARSITIES = [i / 10 for i in range(0, 10)]  # 0.0 .. 0.9


def run_experiment(
    name: str,
    es_config: ESConfig,
    num_epochs: int = 2,
    steps_per_epoch: int = 300,
    batch_size: int = 256,
    num_workers: int = 4,
    extra_model_kwargs: dict | None = None,
) -> dict:
    """Run training for a given ES config and return collected metrics."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    es = EvolutionStrategyCNN(config=es_config, device=device, **(extra_model_kwargs or {}))
    rewards_curve: List[float] = []
    epoch_means: List[float] = []
    val_accs: List[float] = []

    for epoch in range(1, num_epochs + 1):
        rewards = []
        for step, batch in enumerate(train_loader, start=1):
            reward = es.step(batch)
            rewards.append(reward)
            rewards_curve.append(reward)
            if steps_per_epoch and step >= steps_per_epoch:
                break

        epoch_mean = sum(rewards) / max(len(rewards), 1)
        epoch_means.append(epoch_mean)

        val_acc = evaluate_accuracy(es, test_loader)
        val_accs.append(val_acc)
        print(f"[{name}] epoch {epoch} mean_reward={epoch_mean:.4f} val_acc={val_acc:.2f}%")

    return {
        "name": name,
        "config": es_config,
        "rewards": rewards_curve,
        "epoch_means": epoch_means,
        "val_accs": val_accs,
    }


def plot_reward_curves(runs: Sequence[dict], out_path: Path) -> None:
    plt.figure()
    for run in runs:
        plt.plot(run["rewards"], label=run["name"])
    plt.xlabel("Step")
    plt.ylabel("Reward (negative loss)")
    plt.legend()
    plt.tight_layout()
    IMG_DIR.mkdir(exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_val_acc(runs: Sequence[dict], out_path: Path) -> None:
    plt.figure()
    for run in runs:
        epochs = list(range(1, len(run["val_accs"]) + 1))
        plt.plot(epochs, run["val_accs"], marker="o", label=run["name"])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    IMG_DIR.mkdir(exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    runs = []
    for sparsity in RECURRENT_SPARSITIES:
        cfg = ESConfig(npop=NPOP, sigma=SIGMA, alpha=ALPHA)
        name = f"sparsity_{sparsity:.1f}"
        print(
            f"=== Running config: {name} | npop={cfg.npop}, sigma={cfg.sigma}, alpha={cfg.alpha}, "
            f"bag_size={BAG_SIZE}, steps={TIME_STEPS}, recurrent_sparsity={sparsity}"
        )
        runs.append(
            run_experiment(
                name,
                cfg,
                num_epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                extra_model_kwargs={
                    "bag_size": BAG_SIZE,
                    "steps": TIME_STEPS,
                    "recurrent_sparsity": sparsity,
                },
            )
        )

    IMG_DIR.mkdir(exist_ok=True)
    plot_reward_curves(runs, IMG_DIR / "grid_rewards.png")
    plot_val_acc(runs, IMG_DIR / "grid_val_acc.png")

    print("\nFinal validation accuracies:")
    for run in runs:
        if run["val_accs"]:
            print(f"{run['name']}: {run['val_accs'][-1]:.2f}%")


if __name__ == "__main__":
    main()

