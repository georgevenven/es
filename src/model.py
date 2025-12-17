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
        layer_sizes: Optional[Iterable[int]] = None,
        layer_names: Optional[Iterable[str]] = None,
        steps: int = 5,
        anti_hebb_lambda: float = 1e-3,
        recurrent_sparsity: float = 0.9,
        recurrent_groups: int = 1,
    ) -> None:
        super().__init__()
        if layer_sizes is None:
            layer_sizes_list = [int(bag_size)]
        else:
            layer_sizes_list = [int(s) for s in layer_sizes]
        if not layer_sizes_list:
            raise ValueError("layer_sizes must contain at least one layer size")
        if any(s <= 0 for s in layer_sizes_list):
            raise ValueError(f"All layer sizes must be > 0, got {layer_sizes_list}")
        self.layer_sizes = layer_sizes_list
        self.n_layers = len(self.layer_sizes)
        if layer_names is None:
            layer_names_list = [f"L{i+1}" for i in range(self.n_layers)]
        else:
            layer_names_list = [str(n) for n in layer_names]
        if len(layer_names_list) != self.n_layers:
            raise ValueError(
                f"layer_names length ({len(layer_names_list)}) must match layer_sizes length ({self.n_layers})"
            )
        if len(set(layer_names_list)) != len(layer_names_list):
            raise ValueError(f"layer_names must be unique, got {layer_names_list}")
        if "I" in layer_names_list and layer_names_list.index("I") != 0:
            raise ValueError("If provided, 'I' (input layer) must be the first layer")
        self.layer_names = layer_names_list
        self.action_layer_idx = (
            layer_names_list.index("A") if "A" in layer_names_list else (self.n_layers - 1)
        )
        self.class_layer_idx = (
            layer_names_list.index("C") if "C" in layer_names_list else (self.n_layers - 1)
        )
        # For backward compatibility, bag_size refers to the top layer width.
        self.bag_size = int(self.layer_sizes[-1])
        self.steps = steps
        self.anti_hebb_lambda = anti_hebb_lambda
        self.recurrent_sparsity = recurrent_sparsity
        self.recurrent_groups = int(recurrent_groups)
        if self.recurrent_groups < 1:
            raise ValueError(f"recurrent_groups must be >= 1, got {self.recurrent_groups}")
        if self.recurrent_groups > 1:
            for s in self.layer_sizes:
                if s % self.recurrent_groups != 0:
                    raise ValueError(
                        f"layer size ({s}) must be divisible by recurrent_groups ({self.recurrent_groups})"
                    )
            self._block_sizes = [s // self.recurrent_groups for s in self.layer_sizes]
        else:
            self._block_sizes = []
        self.patch_size = 8
        self.image_size = 28
        self.start_row = (self.image_size - self.patch_size) // 2
        self.start_col = (self.image_size - self.patch_size) // 2

        # Patch pixels + (row, col) location of the context window.
        self.input_proj = nn.Linear(self.patch_size * self.patch_size + 2, self.layer_sizes[0])
        self._ff_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        self.ff_projs = nn.ModuleDict()
        # Default: simple chain i -> i+1
        for i in range(self.n_layers - 1):
            j = i + 1
            self._ff_sources[j].append(i)
            self.ff_projs[f"{i}_{j}"] = nn.Linear(self.layer_sizes[i], self.layer_sizes[j])

        # Special case: if both A and C are present, and there is an M* layer before them,
        # feed both A and C from the last M* layer (1-step delayed), rather than chaining.
        if "A" in layer_names_list and "C" in layer_names_list:
            a_idx = self.action_layer_idx
            c_idx = self.class_layer_idx
            m_candidates = [
                i
                for i, nm in enumerate(layer_names_list)
                if nm.upper().startswith("M") and i < a_idx and i < c_idx
            ]
            if m_candidates:
                m_idx = max(m_candidates)
                for dst in (a_idx, c_idx):
                    if dst != 0:
                        self._ff_sources[dst] = [m_idx]
                        key = f"{m_idx}_{dst}"
                        if key not in self.ff_projs:
                            self.ff_projs[key] = nn.Linear(self.layer_sizes[m_idx], self.layer_sizes[dst])

        self.state_norms = nn.ModuleList([nn.LayerNorm(s) for s in self.layer_sizes])
        if self.recurrent_groups == 1:
            self.recurrents = nn.ParameterList(
                [nn.Parameter(torch.empty(s, s)) for s in self.layer_sizes]
            )
            self.recurrent_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(s)) for s in self.layer_sizes]
            )
            # Fixed recurrent connectivity mask (1 = active connection, 0 = masked).
            keep_prob = max(0.0, min(1.0, 1.0 - float(recurrent_sparsity)))
            for i, s in enumerate(self.layer_sizes):
                mask = (torch.rand(s, s) < keep_prob).to(torch.float32)
                self.register_buffer(f"recurrent_mask_{i}", mask)
        else:
            # Block-diagonal recurrence: recurrent weight is represented as G independent blocks.
            # This enables compute proportional to 1/G using dense matmuls (GPU-friendly).
            self.recurrent_blocks_list = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(self.recurrent_groups, bs, bs))
                    for bs in self._block_sizes
                ]
            )
            self.recurrent_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.recurrent_groups, bs)) for bs in self._block_sizes]
            )
        self.controller = nn.Linear(self.layer_sizes[self.action_layer_idx], 2)
        self.out_proj = nn.Linear(self.layer_sizes[self.class_layer_idx], 10)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # He/Kaiming init for layers used with ReLU.
        init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        init.zeros_(self.input_proj.bias)

        if self.recurrent_groups == 1:
            for i, w in enumerate(self.recurrents):
                init.kaiming_normal_(w, nonlinearity="relu")
                # Respect fixed connectivity from the start.
                with torch.no_grad():
                    w.mul_(getattr(self, f"recurrent_mask_{i}"))
            for b in self.recurrent_biases:
                init.zeros_(b)
        else:
            for w in self.recurrent_blocks_list:
                init.kaiming_normal_(w, nonlinearity="relu")
            for b in self.recurrent_biases:
                init.zeros_(b)

        for _, proj in self.ff_projs.items():
            init.kaiming_normal_(proj.weight, nonlinearity="relu")
            init.zeros_(proj.bias)

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
        if return_all:
            return logits_series

        # Aggregate across time by "most present" prediction:
        # take the per-step argmax class, mode across time, then average logits
        # over the timesteps that predicted that mode class.
        logits_t = torch.stack(logits_series, dim=0)  # [T, B, 10]
        preds_t = logits_t.argmax(dim=-1)  # [T, B]
        mode_pred = preds_t.mode(dim=0).values  # [B]
        mask = preds_t.eq(mode_pred.unsqueeze(0)).to(logits_t.dtype)  # [T, B]
        denom = mask.sum(dim=0).clamp(min=1.0)  # [B]
        agg = (logits_t * mask.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)  # [B, 10]
        return agg

    def _compute_move(self, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Convert controller outputs to integer moves (row, col).

        Each controller component is assumed to be in [-1, 1] (e.g. tanh output) and
        is mapped to an integer jump size up to the full valid coordinate range.
        Small magnitudes floor to 0 (no-op).
        """
        max_jump = int(self.image_size - self.patch_size)
        scale = float(max_jump + 1)

        # floor(|ctrl| * (max_jump+1)) gives 0 for small values and max_jump for ctrl≈±1.
        row_steps = torch.floor(ctrl[:, 0].abs() * scale).clamp(max=max_jump).to(torch.int64)
        col_steps = torch.floor(ctrl[:, 1].abs() * scale).clamp(max=max_jump).to(torch.int64)
        row_move = torch.sign(ctrl[:, 0]).to(torch.int64) * row_steps
        col_move = torch.sign(ctrl[:, 1]).to(torch.int64) * col_steps
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

        states: list[Optional[torch.Tensor]] = [None for _ in range(self.n_layers)]
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

            prev_states = states
            new_states: list[Optional[torch.Tensor]] = [None for _ in range(self.n_layers)]

            # Layer 0 consumes the external input each step.
            if prev_states[0] is None:
                new_states[0] = self.state_norms[0](F.relu(in_act))
            else:
                if self.recurrent_groups == 1:
                    recurrent_eff = self.recurrents[0] * getattr(self, "recurrent_mask_0")
                    rec = F.linear(prev_states[0], recurrent_eff, self.recurrent_biases[0])
                else:
                    bs = self._block_sizes[0]
                    st = prev_states[0].view(B, self.recurrent_groups, bs)
                    rec = torch.einsum("bgs,gsh->bgh", st, self.recurrent_blocks_list[0])
                    rec = rec + self.recurrent_biases[0]
                    rec = rec.reshape(B, self.layer_sizes[0])
                new_states[0] = self.state_norms[0](F.relu(in_act + rec))

            # Higher layers: receive input only from their feedforward sources with a 1-step delay.
            for li in range(1, self.n_layers):
                if prev_states[li] is None and all(prev_states[s] is None for s in self._ff_sources[li]):
                    new_states[li] = None
                    continue
                in_terms = []
                for src in self._ff_sources[li]:
                    st = prev_states[src]
                    if st is None:
                        continue
                    in_terms.append(self.ff_projs[f"{src}_{li}"](st))
                in_act_li = sum(in_terms) if in_terms else None

                if prev_states[li] is None:
                    if in_act_li is None:
                        new_states[li] = None
                    else:
                        new_states[li] = self.state_norms[li](F.relu(in_act_li))
                else:
                    if in_act_li is None:
                        # No feedforward input this step; only recurrent update.
                        in_act_li = torch.zeros_like(prev_states[li])
                    if self.recurrent_groups == 1:
                        recurrent_eff = self.recurrents[li] * getattr(self, f"recurrent_mask_{li}")
                        rec = F.linear(prev_states[li], recurrent_eff, self.recurrent_biases[li])
                    else:
                        bs = self._block_sizes[li]
                        st = prev_states[li].view(B, self.recurrent_groups, bs)
                        rec = torch.einsum("bgs,gsh->bgh", st, self.recurrent_blocks_list[li])
                        rec = rec + self.recurrent_biases[li]
                        rec = rec.reshape(B, self.layer_sizes[li])
                    new_states[li] = self.state_norms[li](F.relu(in_act_li + rec))

            states = new_states
            class_state = states[self.class_layer_idx]
            if class_state is None:
                class_state_for_readout = torch.zeros(
                    B,
                    self.layer_sizes[self.class_layer_idx],
                    device=device,
                    dtype=in_act.dtype,
                )
            else:
                class_state_for_readout = class_state

            logits = self.out_proj(class_state_for_readout)
            logits_series.append(logits)
            if record_positions:
                positions_hist.append(positions.detach().cpu())
            if record_states:
                states_hist.append(class_state_for_readout.detach().cpu())

            action_state = states[self.action_layer_idx]
            if action_state is None:
                move = torch.zeros(B, 2, device=device, dtype=torch.int64)
            else:
                ctrl = torch.tanh(self.controller(action_state))
                move = self._compute_move(ctrl)
            positions = positions + move
            positions = positions.clamp(0, self.image_size - self.patch_size)

        return logits_series, positions_hist, states_hist


@dataclass
class ESConfig:
    npop: int = 250
    sigma: float = 0.1
    alpha: float = 1e-2
    weight_decay: float = 0.0
    amp: bool = False


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
        layer_sizes: Optional[Iterable[int]] = None,
        layer_names: Optional[Iterable[str]] = None,
        steps: int = 5,
        anti_hebb_lambda: float = 1e-3,
        recurrent_sparsity: float = 0.9,
        recurrent_groups: int = 1,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_devices: Tuple[torch.device, ...] = eval_devices or (self.device,)

        self.model = CyclicNet(
            bag_size=bag_size,
            layer_sizes=layer_sizes,
            layer_names=layer_names,
            steps=steps,
            anti_hebb_lambda=anti_hebb_lambda,
            recurrent_sparsity=recurrent_sparsity,
            recurrent_groups=recurrent_groups,
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

            use_amp = bool(self.config.amp) and self.device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
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

            use_amp = bool(self.config.amp) and dev_inputs.device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
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
        delta_es = (cfg.alpha / cfg.sigma) * step_dir
        # L2 weight decay (SGD-style): w <- w + delta_es - alpha*wd*w
        # (keeps weight norms from drifting upwards unchecked)
        delta = delta_es - (cfg.alpha * cfg.weight_decay) * w
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
                "weight_norm": float(w_norm.item()),
                "delta_norm": float(delta_norm.item()),
                "delta_rel": float((delta_norm / (w_norm + 1e-12)).item()),
                "step_dir_norm": float(step_dir_norm.item()),
                "snr_corr": float(snr_corr.item()),
                "delta_cos_prev": float(cos_prev.item()),
            }

        return rewards_mean.item()

    def parameters(self):
        return self.model.parameters()

