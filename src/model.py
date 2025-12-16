"""Cyclical recurrent model and Evolution Strategy optimizer for MNIST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.func import functional_call, vmap

__all__ = ["CyclicNet", "EvolutionStrategyCNN"]


class CyclicNet(nn.Module):
    """
    A small recurrent "bag of neurons" with all-to-all connectivity.

    Input is injected for one step, state propagates for T steps, and the final
    state is read out as logits.
    """

    def __init__(
        self,
        bag_size: int = 15,
        steps: int = 5,
        anti_hebb_lambda: float = 1e-3,
        recurrent_sparsity: float = 0.9,
    ) -> None:
        super().__init__()
        self.bag_size = bag_size
        self.steps = steps
        self.anti_hebb_lambda = anti_hebb_lambda
        self.recurrent_sparsity = recurrent_sparsity
        self.patch_size = 8
        self.image_size = 28
        self.start_row = (self.image_size - self.patch_size) // 2
        self.start_col = (self.image_size - self.patch_size) // 2

        # Patch pixels + (row, col) location of the context window.
        self.input_proj = nn.Linear(self.patch_size * self.patch_size + 2, bag_size)
        self.state_norm = nn.LayerNorm(bag_size)
        self.recurrent = nn.Parameter(torch.empty(bag_size, bag_size))
        self.recurrent_bias = nn.Parameter(torch.zeros(bag_size))
        # Fixed recurrent connectivity mask (1 = active connection, 0 = masked).
        keep_prob = max(0.0, min(1.0, 1.0 - float(recurrent_sparsity)))
        mask = (torch.rand(bag_size, bag_size) < keep_prob).to(torch.float32)
        self.register_buffer("recurrent_mask", mask)
        self.controller = nn.Linear(bag_size, 2)
        self.out_proj = nn.Linear(bag_size, 10)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # He/Kaiming init for layers used with ReLU.
        init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        init.zeros_(self.input_proj.bias)

        init.kaiming_normal_(self.recurrent, nonlinearity="relu")
        # Respect fixed connectivity from the start.
        with torch.no_grad():
            self.recurrent.mul_(self.recurrent_mask)
        init.zeros_(self.recurrent_bias)

        init.kaiming_normal_(self.controller.weight, nonlinearity="relu")
        init.zeros_(self.controller.bias)

        init.kaiming_normal_(self.out_proj.weight, nonlinearity="relu")
        init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        logits_series, _, _ = self._rollout(
            x, steps_override=None, record_positions=False, start_positions=start_positions
        )
        return logits_series if return_all else logits_series[-1]

    def _compute_move(self, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Convert controller outputs to integer moves (up/down/left/right).
        Chooses the axis with larger magnitude; step is sign {-1, 0, 1}.
        """
        mag_row = ctrl[:, 0].abs()
        mag_col = ctrl[:, 1].abs()
        row_mask = mag_row >= mag_col

        row_move = torch.where(
            row_mask, torch.sign(ctrl[:, 0]).to(torch.int64), torch.zeros_like(ctrl[:, 0], dtype=torch.int64)
        )
        col_move = torch.where(
            row_mask, torch.zeros_like(ctrl[:, 1], dtype=torch.int64), torch.sign(ctrl[:, 1]).to(torch.int64)
        )
        return torch.stack([row_move, col_move], dim=1)

    def _rollout(
        self,
        x: torch.Tensor,
        steps_override: Optional[int] = None,
        record_positions: bool = False,
        record_states: bool = False,
        start_positions: Optional[torch.Tensor] = None,
    ):
        """
        Run the recurrent dynamics over a sequence of glimpses.

        Args:
            x: input images [B, 1, 28, 28]
            steps_override: optional number of steps; defaults to self.steps
            record_positions: if True, returns position history per step

        Returns:
            logits_series: list of [B, 10] tensors for each step
            positions_hist: list of [B, 2] (row, col) if requested else None
            states_hist: list of [B, bag_size] if requested else None
        """
        B = x.shape[0]
        device = x.device
        steps = steps_override or self.steps
        patch_dim = self.patch_size * self.patch_size
        stride = 1
        patches_per_side = self.image_size - self.patch_size + 1
        unfolded = F.unfold(x, kernel_size=self.patch_size, stride=stride)  # [B, patch_dim, n_patches]

        if start_positions is None:
            positions = torch.tensor(
                [self.start_row, self.start_col], device=device, dtype=torch.int64
            ).repeat(B, 1)
        else:
            positions = start_positions.to(device=device, dtype=torch.int64)
        positions_hist = [] if record_positions else None
        states_hist = [] if record_states else None

        state = None
        logits_series = []
        for _ in range(steps):
            # Select patches via unfold + gather to stay vmap-friendly
            patch_idx = positions[:, 0] * patches_per_side + positions[:, 1]  # [B]
            gather_idx = patch_idx.view(B, 1, 1).expand(B, patch_dim, 1)
            flat = unfolded.gather(2, gather_idx).squeeze(2)  # [B, patch_dim]

            # Feed location back into the bag input.
            # Normalize to [-1, 1] over the valid top-left coordinate range.
            denom = max(self.image_size - self.patch_size, 1)
            pos = (positions.to(flat.dtype) / float(denom)) * 2.0 - 1.0  # [B, 2]
            in_act = self.input_proj(torch.cat([flat, pos], dim=1))

            if state is None:
                state = self.state_norm(F.relu(in_act))
            else:
                # Anti-hebbian decay within the bag: reduce connections for co-active pairs.
                coact = torch.einsum("bi,bj->ij", state, state) / max(B, 1)
                recurrent_eff = (self.recurrent - self.anti_hebb_lambda * coact) * self.recurrent_mask
                state = self.state_norm(
                    F.relu(in_act + F.linear(state, recurrent_eff, self.recurrent_bias))
                )

            logits = self.out_proj(state)
            logits_series.append(logits)
            if record_positions:
                positions_hist.append(positions.detach().cpu())
            if record_states:
                states_hist.append(state.detach().cpu())

            ctrl = torch.tanh(self.controller(state))
            move = self._compute_move(ctrl)
            positions = positions + move
            positions = positions.clamp(0, self.image_size - self.patch_size)

        return logits_series, positions_hist, states_hist


@dataclass
class ESConfig:
    npop: int = 250
    sigma: float = 0.1
    alpha: float = 1e-2


class EvolutionStrategyCNN:
    """
    Evolution Strategies optimizer wrapped around a small CNN.

    The ES update mirrors the provided numpy snippet but operates on the
    flattened parameter vector of the CNN and uses a batch reward computed from
    cross-entropy loss on MNIST batches.
    """

    def __init__(
        self,
        config: ESConfig = ESConfig(),
        device: Optional[torch.device] = None,
        *,
        eval_devices: Optional[Tuple[torch.device, ...]] = None,
        bag_size: int = 15,
        steps: int = 5,
        anti_hebb_lambda: float = 1e-3,
        recurrent_sparsity: float = 0.9,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_devices: Tuple[torch.device, ...] = eval_devices or (self.device,)

        self.model = CyclicNet(
            bag_size=bag_size,
            steps=steps,
            anti_hebb_lambda=anti_hebb_lambda,
            recurrent_sparsity=recurrent_sparsity,
        ).to(self.device)
        self._w = parameters_to_vector(self.model.parameters()).detach()
        self._param_shapes = [
            (name, p.shape, p.numel()) for name, p in self.model.named_parameters()
        ]
        self._buffers = dict(self.model.named_buffers())
        self._buffers_by_device: Dict[torch.device, Dict[str, torch.Tensor]] = {}
        self.last_step_metrics: Dict[str, float] = {}
        self._prev_delta: Optional[torch.Tensor] = None

        for dev in self.eval_devices:
            self._buffers_by_device[dev] = {
                k: v.to(dev, non_blocking=True) for k, v in self._buffers.items()
            }

    @property
    def param_size(self) -> int:
        return self._w.numel()

    @torch.no_grad()
    def evaluate(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Compute reward (negative loss) for current weights on a batch."""
        self.model.eval()
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        return -loss.item()

    def _vector_to_param_dict(self, vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat vector to a parameter dict matching model structure."""
        param_dict = {}
        idx = 0
        for name, shape, numel in self._param_shapes:
            param_dict[name] = vec[idx : idx + numel].view(shape)
            idx += numel
        return param_dict

    @torch.no_grad()
    def _batched_rewards(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        vectors: torch.Tensor,
        start_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute rewards for many parameter vectors in parallel using vmap.

        Args:
            batch: (images, labels)
            vectors: Tensor of shape [pop, param_dim]
        Returns:
            rewards tensor of shape [pop]
        """
        # If multiple eval devices are configured, split the population across them.
        if len(self.eval_devices) == 1:
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            sp = start_positions.to(self.device, non_blocking=True) if start_positions is not None else None
            buffers = self._buffers_by_device[self.device]

            def single_reward(vec: torch.Tensor) -> torch.Tensor:
                params = self._vector_to_param_dict(vec)
                merged = {**params, **buffers}
                logits = functional_call(self.model, merged, (inputs, sp))
                loss = F.cross_entropy(logits, targets, reduction="mean")
                return -loss

            return vmap(single_reward)(vectors)

        inputs_cpu, targets_cpu = batch
        pop = vectors.shape[0]
        ndev = len(self.eval_devices)
        chunks = torch.chunk(vectors, ndev, dim=0)
        rewards_parts = []

        # Move batch once per device.
        per_dev_batch = []
        for dev in self.eval_devices:
            per_dev_batch.append(
                (
                    inputs_cpu.to(dev, non_blocking=True),
                    targets_cpu.to(dev, non_blocking=True),
                    start_positions.to(dev, non_blocking=True) if start_positions is not None else None,
                    self._buffers_by_device[dev],
                )
            )

        for chunk, (dev_inputs, dev_targets, dev_sp, dev_buffers) in zip(chunks, per_dev_batch):
            if chunk.numel() == 0:
                continue
            dev_vecs = chunk.to(dev_inputs.device, non_blocking=True)

            def single_reward(vec: torch.Tensor) -> torch.Tensor:
                params = self._vector_to_param_dict(vec)
                merged = {**params, **dev_buffers}
                logits = functional_call(self.model, merged, (dev_inputs, dev_sp))
                loss = F.cross_entropy(logits, dev_targets, reduction="mean")
                return -loss

            r = vmap(single_reward)(dev_vecs)
            rewards_parts.append(r.to(self.device, non_blocking=True))

        rewards = torch.cat(rewards_parts, dim=0)
        # torch.chunk keeps order; zip preserves order, but if pop isn't divisible by ndev,
        # some chunks may be empty on later devices; concatenation still preserves order.
        if rewards.shape[0] != pop:
            raise RuntimeError(f"Expected {pop} rewards, got {rewards.shape[0]}")
        return rewards

    @torch.no_grad()
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Perform one ES update using a single batch.

        Returns:
            Average reward across the sampled population.
        """
        cfg = self.config
        w = self._w
        noise = torch.randn(cfg.npop, w.numel(), device=self.device)
        candidates = w + cfg.sigma * noise

        # Batched rewards via functional_call + vmap
        # Start context window at the center (default behavior).
        start_positions = None
        rewards = self._batched_rewards(batch, candidates, start_positions)

        # Normalize rewards
        rewards_mean = rewards.mean()
        rewards_std = rewards.std(unbiased=False) + 1e-8
        A = (rewards - rewards_mean) / rewards_std

        # ES parameter update
        step_dir = (A.unsqueeze(1) * noise).mean(dim=0)
        delta = (cfg.alpha / cfg.sigma) * step_dir
        w = w + delta

        self._w = w
        vector_to_parameters(w, self.model.parameters())

        # Diagnostics for logging/debugging (no effect on update)
        with torch.no_grad():
            w_norm = w.norm()
            delta_norm = delta.norm()

            # Correlation between standardized rewards A and projection of noise onto step_dir
            step_dir_norm = step_dir.norm()
            denom = step_dir_norm + 1e-12
            proj = (noise @ step_dir) / denom  # [pop]
            proj_std = proj.std(unbiased=False)
            A_std = A.std(unbiased=False)
            if proj_std.item() > 0.0 and A_std.item() > 0.0:
                snr_corr = ((A - A.mean()) * (proj - proj.mean())).mean() / (A_std * proj_std + 1e-12)
            else:
                snr_corr = torch.tensor(0.0, device=self.device)

            if self._prev_delta is not None:
                cos_prev = (delta @ self._prev_delta) / (delta_norm * (self._prev_delta.norm()) + 1e-12)
            else:
                cos_prev = torch.tensor(0.0, device=self.device)
            self._prev_delta = delta.detach()

            self.last_step_metrics = {
                "reward_mean": float(rewards_mean.item()),
                "reward_std": float((rewards_std - 1e-8).item()),
                "reward_min": float(rewards.min().item()),
                "reward_med": float(rewards.median().item()),
                "reward_max": float(rewards.max().item()),
                "delta_norm": float(delta_norm.item()),
                "delta_rel": float((delta_norm / (w_norm + 1e-12)).item()),
                "step_dir_norm": float(step_dir_norm.item()),
                "snr_corr": float(snr_corr.item()),
                "delta_cos_prev": float(cos_prev.item()),
            }

        return rewards_mean.item()

    def parameters(self):
        return self.model.parameters()

