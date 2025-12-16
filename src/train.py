"""Training loop for CNN + Evolution Strategies on MNIST with visualization."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import random
import argparse

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from data_loader import get_mnist_dataloaders
from model import ESConfig, EvolutionStrategyCNN


def _unnormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo MNIST normalization."""
    mean, std = 0.1307, 0.3081
    return images * std + mean


@torch.no_grad()
def save_prediction_grid(
    es: EvolutionStrategyCNN,
    batch,
    out_path: Path,
    cols: int = 3,
) -> None:
    """
    Save a grid (default 3x3) of MNIST images with predicted labels drawn under each.
    """
    images, _ = batch
    images = images.to(es.device)
    preds = es.model(images).argmax(dim=1).cpu()
    images = _unnormalize(images).cpu()

    images = images[: cols * cols]
    preds = preds[: cols * cols]

    # Dimensions
    pad = 6
    img_size = images.shape[-1]
    text_height = 12
    grid_w = cols * img_size + (cols + 1) * pad
    rows = (len(images) + cols - 1) // cols
    grid_h = rows * (img_size + text_height) + (rows + 1) * pad

    canvas = Image.new("L", (grid_w, grid_h), color=255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (img, pred) in enumerate(zip(images, preds)):
        r, c = divmod(idx, cols)
        x0 = pad + c * (img_size + pad)
        y0 = pad + r * (img_size + text_height + pad)

        arr = (img.squeeze().clamp(0, 1).numpy() * 255).astype("uint8")
        tile = Image.fromarray(arr, mode="L")
        canvas.paste(tile, (x0, y0))

        text = f"pred: {pred.item()}"
        # textbbox provides width/height without relying on deprecated textsize
        text_box = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
        tx = x0 + (img_size - text_w) // 2
        ty = y0 + img_size + 2
        draw.text((tx, ty), text, fill=0, font=font)

    canvas.save(out_path)


@torch.no_grad()
def evaluate_accuracy(es: EvolutionStrategyCNN, loader) -> float:
    """Compute classification accuracy (%) on a loader."""
    es.model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(es.device, non_blocking=True)
        labels = labels.to(es.device, non_blocking=True)
        # Start context window at the center (default behavior).
        preds = es.model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def simulate_dynamics(es: EvolutionStrategyCNN, image: torch.Tensor, steps: int | None = None):
    """
    Roll out the CyclicNet dynamics for a single image.

    Returns a list of logits at each time step (including t=0 after input injection).
    """
    model = es.model
    model.eval()
    steps = steps or getattr(model, "steps", 1)

    logits_series, positions, _ = model._rollout(
        image.unsqueeze(0).to(es.device),
        steps_override=steps,
        record_positions=True,
    )
    # detach to cpu for downstream plotting
    logits_series = [l.squeeze(0).cpu() for l in logits_series]
    positions = [p.squeeze(0).tolist() for p in positions] if positions is not None else None
    return logits_series, positions


@torch.no_grad()
def simulate_sequence(
    es: EvolutionStrategyCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    steps_per_digit: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Roll out dynamics over a sequence of images, switching digits between segments.

    Args:
        images: Tensor [N, 1, 28, 28]
        steps_per_digit: number of steps to let the state evolve per image

    Returns:
        probs_history: Tensor [T, 10] of softmax probabilities over time
        label_track: Tensor [T] of which label was shown at each step
    """
    model = es.model
    model.eval()

    probs_list = []
    label_track = []
    for img, lbl in zip(images, labels):
        logits_series, _, _ = model._rollout(
            img.unsqueeze(0).to(es.device),
            steps_override=steps_per_digit,
            record_positions=False,
        )
        for logits in logits_series:
            probs = torch.softmax(logits.squeeze(0), dim=0).cpu()
            probs_list.append(probs)
            label_track.append(int(lbl))

    probs_history = torch.stack(probs_list, dim=0) if probs_list else torch.empty(0, 10)
    label_track = torch.tensor(label_track, dtype=torch.long)
    return probs_history, label_track


def save_dynamics_gif(es: EvolutionStrategyCNN, image: torch.Tensor, out_path: Path, steps: int | None = None):
    """Create a GIF showing logits evolution over time for one image."""
    logits_series, _ = simulate_dynamics(es, image, steps)
    img = _unnormalize(image).squeeze().cpu().clamp(0, 1)
    base = (img.numpy() * 255).astype("uint8")

    frames = []
    font = ImageFont.load_default()
    pad = 6
    for t, logits in enumerate(logits_series):
        probs = torch.softmax(logits, dim=0)
        pred = probs.argmax().item()

        # Create canvas
        canvas = Image.new("RGB", (200, 80), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Paste digit
        digit = Image.fromarray(base, mode="L").resize((56, 56))
        canvas.paste(digit.convert("RGB"), (pad, pad))

        # Text block
        y = pad
        draw.text((80, y), f"t={t}", fill=(0, 0, 0), font=font)
        y += 14
        draw.text((80, y), f"pred={pred}", fill=(0, 0, 0), font=font)
        y += 14
        top3 = torch.topk(probs, k=3)
        for k_idx in range(top3.indices.numel()):
            cls = top3.indices[k_idx].item()
            p = top3.values[k_idx].item() * 100
            draw.text((80, y), f"{cls}: {p:4.1f}%", fill=(0, 0, 0), font=font)
            y += 12

        frames.append(canvas)

    out_path.parent.mkdir(exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0,
    )


@torch.no_grad()
def save_attention_gif(
    es: EvolutionStrategyCNN,
    images: torch.Tensor,
    out_path: Path,
    steps: int | None = None,
    grid_dim: int = 9,
) -> None:
    """Visualize moving 4x4 windows over many samples at once."""
    n = min(grid_dim * grid_dim, images.shape[0])
    imgs = images[:n]
    # Run rollout per image to get positions and logits
    rollouts = []
    for img in imgs:
        logits_series, positions = simulate_dynamics(es, img, steps)
        rollouts.append((logits_series, positions))
    if not rollouts:
        return

    steps_total = len(rollouts[0][0])
    patch = getattr(es.model, "patch_size", 4)
    img_size = imgs.shape[-1]
    grid_w = grid_dim * img_size
    grid_h = grid_dim * img_size
    pad = 4

    frames = []
    font = ImageFont.load_default()
    # Precompute unnormalized images
    bases = (_unnormalize(imgs).cpu().clamp(0, 1).numpy() * 255).astype("uint8")[:, 0, :, :]

    for t in range(steps_total):
        canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for idx, (logits_series, positions) in enumerate(rollouts):
            if t >= len(logits_series) or positions is None:
                continue
            r, c = positions[t]
            row = idx // grid_dim
            col = idx % grid_dim
            x0 = col * img_size
            y0 = row * img_size
            digit = Image.fromarray(bases[idx], mode="L").convert("RGB")
            canvas.paste(digit, (x0, y0))
            draw.rectangle(
                [x0 + c, y0 + r, x0 + c + patch - 1, y0 + r + patch - 1],
                outline="red",
                width=1,
            )
            probs = torch.softmax(torch.tensor(logits_series[t]), dim=0)
            pred = int(torch.argmax(probs).item())
            draw.text((x0 + 2, y0 + 2), f"{pred}", fill="red", font=font)

        draw.text((grid_w - 40, grid_h - 14), f"t={t}", fill="black", font=font)
        frames.append(canvas)

    out_path.parent.mkdir(exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
    )


@torch.no_grad()
def save_prob_lines_gif(
    es: EvolutionStrategyCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    out_path: Path,
    steps_per_digit: int = 6,
) -> None:
    """
    Create a GIF showing probability trajectories as digits switch.
    """
    probs_history, label_track = simulate_sequence(es, images, labels, steps_per_digit)
    if probs_history.numel() == 0:
        return

    total_steps = probs_history.shape[0]
    frames = []
    colors = plt.cm.tab10.colors

    # Precompute unnormalized digits for display
    digits = _unnormalize(images).cpu().clamp(0, 1)

    step_idx = 0
    for digit_idx, (img, lbl) in enumerate(zip(digits, labels)):
        for _ in range(steps_per_digit):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))

            # Left: digit
            axes[0].imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[0].axis("off")
            axes[0].set_title(f"digit {int(lbl)}")

            # Right: probability lines
            t = step_idx
            for cls in range(10):
                axes[1].plot(
                    range(t + 1),
                    probs_history[: t + 1, cls],
                    color=colors[cls % len(colors)],
                    linewidth=1,
                    alpha=0.8,
                )
            axes[1].axvline(t, color="k", linestyle="--", linewidth=0.7, alpha=0.6)
            axes[1].set_ylim(0, 1)
            axes[1].set_xlabel("time")
            axes[1].set_ylabel("P(class)")
            axes[1].set_title(f"t={t}")

            # annotate current prediction
            probs_t = probs_history[t]
            pred = int(torch.argmax(probs_t).item())
            axes[1].text(
                0.02,
                0.95,
                f"pred={pred} ({probs_t[pred]*100:.1f}%)",
                transform=axes[1].transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frame = Image.open(buf).convert("RGB")
            frames.append(frame)

            step_idx += 1
            if step_idx >= total_steps:
                break

    out_path.parent.mkdir(exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0,
    )


@torch.no_grad()
def save_activation_pca_gif(
    es: EvolutionStrategyCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    out_path: Path,
    steps: int | None = None,
    max_points: int = 256,
) -> None:
    """
    Create a GIF of 2D PCA of the bag (hidden state) over recurrent time steps.

    Each frame t shows the batch's hidden states projected to 2D, colored by label.
    PCA is fit on the concatenation of all time steps to keep axes consistent.
    """
    model = es.model
    model.eval()
    steps = steps or getattr(model, "steps", 1)

    n = min(int(max_points), int(images.shape[0]))
    imgs = images[:n].to(es.device)
    lbls = labels[:n].cpu()

    _, _, states_hist = model._rollout(
        imgs,
        steps_override=steps,
        record_positions=False,
        record_states=True,
    )
    if not states_hist:
        return

    # X: [steps*n, bag_size]
    X = torch.cat(states_hist, dim=0).to(torch.float32)  # already on CPU from _rollout
    Xc = X - X.mean(dim=0, keepdim=True)
    # PCA via SVD: Xc = U S Vh, components are rows of Vh.
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vh[:2, :]  # [2, bag_size]
    Z = Xc @ comps.T  # [steps*n, 2]

    # Fixed axis limits across frames for stable animation
    zmin = Z.min(dim=0).values
    zmax = Z.max(dim=0).values

    # Colors for digits 0-9
    colors = plt.cm.tab10.colors
    frames = []

    for t in range(len(states_hist)):
        Zt = Z[t * n : (t + 1) * n]

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        for cls in range(10):
            mask = (lbls == cls)
            if int(mask.sum().item()) == 0:
                continue
            pts = Zt[mask]
            ax.scatter(
                pts[:, 0].numpy(),
                pts[:, 1].numpy(),
                s=12,
                alpha=0.85,
                color=colors[cls % len(colors)],
                label=str(cls),
            )

        ax.set_xlim(float(zmin[0]), float(zmax[0]))
        ax.set_ylim(float(zmin[1]), float(zmax[1]))
        ax.set_title(f"Bag activations PCA (t={t})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    out_path.parent.mkdir(exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0,
    )


def train(
    num_epochs: int = 10,
    steps_per_epoch: int = 500,
    batch_size: int = 256,
    num_workers: int = 8,
    time_steps: int = 5,
    bag_size: int = 15,
    recurrent_sparsity: float = 0.9,
    devices: str = "cuda:0",
    npop: int = ESConfig.npop,
    sigma: float = ESConfig.sigma,
    alpha: float = ESConfig.alpha,
) -> EvolutionStrategyCNN:
    """
    Run ES training over MNIST using a single GPU if available.

    Args:
        num_epochs: Number of training epochs.
        steps_per_epoch: Max batches per epoch (limits runtime).
        batch_size: Batch size for MNIST loader.
        num_workers: Dataloader workers.
    """
    if torch.cuda.is_available():
        dev_list = [d.strip() for d in devices.split(",") if d.strip()]
        eval_devices = tuple(torch.device(d) for d in (dev_list or ["cuda:0"]))
        device = eval_devices[0]
    else:
        eval_devices = (torch.device("cpu"),)
        device = torch.device("cpu")

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    es_config = ESConfig(npop=npop, sigma=sigma, alpha=alpha)
    es = EvolutionStrategyCNN(
        config=es_config,
        device=device,
        eval_devices=eval_devices,
        steps=time_steps,
        bag_size=bag_size,
        recurrent_sparsity=recurrent_sparsity,
    )
    viz_batch = None
    test_batch = None

    for epoch in range(1, num_epochs + 1):
        rewards = []
        for step, batch in enumerate(train_loader, start=1):
            if viz_batch is None:
                # Keep a small batch on CPU for visualization reuse.
                images, labels = batch
                viz_batch = (images[:9].cpu(), labels[:9].cpu())
            if test_batch is None:
                test_batch = next(iter(test_loader))

            reward = es.step(batch)
            rewards.append(reward)

            if step % 50 == 0:
                avg = sum(rewards[-50:]) / len(rewards[-50:])
                m = getattr(es, "last_step_metrics", {}) or {}
                print(
                    "[epoch {epoch} step {step}] "
                    "reward_avg={avg:.4f} "
                    "pop(mean/std/min/med/max)={rmean:.4f}/{rstd:.4f}/{rmin:.4f}/{rmed:.4f}/{rmax:.4f} "
                    "delta_norm={dn:.3e} delta_rel={dr:.3e} "
                    "step_dir_norm={sdn:.3e} snr_corr={sc:.3f} cos_prev={cp:.3f}".format(
                        epoch=epoch,
                        step=step,
                        avg=avg,
                        rmean=m.get("reward_mean", float("nan")),
                        rstd=m.get("reward_std", float("nan")),
                        rmin=m.get("reward_min", float("nan")),
                        rmed=m.get("reward_med", float("nan")),
                        rmax=m.get("reward_max", float("nan")),
                        dn=m.get("delta_norm", float("nan")),
                        dr=m.get("delta_rel", float("nan")),
                        sdn=m.get("step_dir_norm", float("nan")),
                        sc=m.get("snr_corr", float("nan")),
                        cp=m.get("delta_cos_prev", float("nan")),
                    )
                )

            if steps_per_epoch and step >= steps_per_epoch:
                break

        epoch_avg = sum(rewards) / len(rewards)
        print(f"Epoch {epoch} complete: mean_reward={epoch_avg:.4f}")

        val_acc = evaluate_accuracy(es, test_loader)
        print(f"Validation accuracy: {val_acc:.2f}%")

        if viz_batch is not None:
            project_root = Path(__file__).resolve().parent.parent
            out_dir = project_root / "imgs"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"epoch_{epoch:03d}.png"
            save_prediction_grid(es, viz_batch, out_path)
            print(f"Saved prediction grid to {out_path}")

    if test_batch is not None:
        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / "imgs"
        out_dir.mkdir(exist_ok=True)
        imgs = test_batch[0].cpu()
        lbls = test_batch[1].cpu()
        gif_path = out_dir / "dynamics.gif"
        save_dynamics_gif(es, imgs[0], gif_path)
        print(f"Saved dynamics GIF to {gif_path}")

        attn_gif_path = out_dir / "attention.gif"
        # Build a 10x10 collage: column = digit (0..9), 10 samples per digit from val set.
        per_digit: list[list[torch.Tensor]] = [[] for _ in range(10)]
        for batch_imgs, batch_lbls in test_loader:
            batch_imgs = batch_imgs.cpu()
            batch_lbls = batch_lbls.cpu()
            for img, lbl in zip(batch_imgs, batch_lbls):
                d = int(lbl.item())
                if 0 <= d <= 9 and len(per_digit[d]) < 10:
                    per_digit[d].append(img)
            if all(len(per_digit[d]) >= 10 for d in range(10)):
                break

        ordered = []
        for row in range(10):
            for digit in range(10):
                ordered.append(per_digit[digit][row])
        attn_imgs = torch.stack(ordered, dim=0)

        save_attention_gif(es, attn_imgs, attn_gif_path, steps=time_steps, grid_dim=10)
        print(f"Saved attention GIF to {attn_gif_path}")

        # Probability trajectories with digit switches
        num_seq = min(4, imgs.shape[0])
        idx = torch.randperm(imgs.shape[0])[:num_seq]
        seq_imgs = imgs[idx]
        seq_lbls = lbls[idx]
        line_gif_path = out_dir / "dynamics_probs.gif"
        save_prob_lines_gif(es, seq_imgs, seq_lbls, line_gif_path, steps_per_digit=6)
        print(f"Saved probability dynamics GIF to {line_gif_path}")

        pca_gif_path = out_dir / "pca_activations.gif"
        save_activation_pca_gif(es, imgs, lbls, pca_gif_path, steps=time_steps, max_points=256)
        print(f"Saved activation PCA GIF to {pca_gif_path}")

    return es


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ES CyclicNet on MNIST.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--steps-per-epoch", type=int, default=500, help="Batches per epoch (0 for full epoch)."
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--time-steps", type=int, default=5, help="Recurrent time steps for CyclicNet.")
    parser.add_argument("--bag-size", type=int, default=15, help="Number of neurons in the bag.")
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda:0",
        help="Comma-separated CUDA devices to split ES population eval across (e.g. cuda:0,cuda:1).",
    )
    parser.add_argument(
        "--recurrent-sparsity",
        type=float,
        default=0.9,
        help="Fraction of recurrent connections masked out (e.g. 0.9 keeps ~10%).",
    )
    parser.add_argument("--npop", type=int, default=ESConfig.npop, help="ES population size.")
    parser.add_argument("--sigma", type=float, default=ESConfig.sigma, help="ES noise std.")
    parser.add_argument("--alpha", type=float, default=ESConfig.alpha, help="ES learning rate.")
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_steps=args.time_steps,
        bag_size=args.bag_size,
        recurrent_sparsity=args.recurrent_sparsity,
        devices=args.devices,
        npop=args.npop,
        sigma=args.sigma,
        alpha=args.alpha,
    )

