# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

import math
from functools import wraps
from typing import Any, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class SchedulerRLMixin:
    @staticmethod
    def require_rollout_enabled_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._require_rollout_enabled()
            return func(self, *args, **kwargs)

        return wrapper

    def release_rollout_resources(self) -> None:
        """Release rollout-owned resources (e.g. noise buffer). Call when denoising ends."""
        if hasattr(self, "_rollout_noise_buffer"):
            self._rollout_noise_buffer = None
        self._rollout_local_log_prob_sum.clear()
        self._rollout_local_log_prob_count.clear()

    def reset_rollout_states(self):
        """Reset rollout states, should be called at the beginning of each new request"""
        self._rollout_enabled = False
        self.release_rollout_resources()

    def prepare_rollout(
        self,
        *,
        noise_level: float = 0.7,
        sde_type: str = "sde",
        log_prob_no_const: bool = False,
        noise_full_shape: Optional[tuple] = None,
    ) -> None:
        """Enable rollout and set SDE/CPS params. Call once before the denoising loop.

        noise_full_shape: When using sequence parallelism (SP), pass the unsharded
            latent shape so variance noise is generated for the full tensor then
            sharded per rank (keeps generator deterministic). Omit or pass None for
            single-rank or when the pipeline will not shard latents; default None
            is correct and does not need to be set by API users.
        """
        self._rollout_enabled = True
        self._rollout_local_log_prob_sum = []
        self._rollout_local_log_prob_count = []
        # Prepare params needed for rollout
        self._rollout_param_log_prob_no_const = log_prob_no_const
        self._rollout_param_noise_level = float(noise_level)
        self._rollout_param_sde_type = sde_type
        self._rollout_param_noise_full_shape = noise_full_shape

        # Prepare extra parameters for sampling
        self._rollout_sigma_max = self.sigmas[min(1, len(self.sigmas) - 1)].item()
    
    def already_prepared_rollout(self):
        return getattr(self, "_rollout_enabled", False)

    def _require_rollout_enabled(self) -> None:
        if not getattr(self, "_rollout_enabled", False):
            raise RuntimeError("prepare_rollout() not called before rollout")

    def _get_or_create_rollout_noise_buffer(
        self,
        full_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create the reusable full-shape noise buffer for SP rollout."""
        buffer = getattr(self, "_rollout_noise_buffer", None)
        if (
            buffer is None
            or buffer.shape != full_shape
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            buffer = torch.empty(full_shape, device=device, dtype=dtype)
            self._rollout_noise_buffer = buffer
        return buffer

    def _rollout_variance_noise(
        self,
        model_output: torch.FloatTensor,
        generator: torch.Generator,
    ) -> torch.FloatTensor:
        """Generate variance noise for rollout. If SP and noise_full_shape given, generate full then shard."""
        noise_full_shape = getattr(self, "_rollout_param_noise_full_shape", None)
        sp_size = get_sp_world_size()
        if sp_size <= 1 or noise_full_shape is None:
            return randn_tensor(
                model_output.shape,
                generator=generator,
                device=get_local_torch_device(),
                dtype=model_output.dtype,
            )
        full_shape = tuple(noise_full_shape)
        local_shape = model_output.shape
        # Infer shard dim: unique d where full_shape[d] == local_shape[d] * sp_size
        shard_dims = [
            d
            for d in range(len(full_shape))
            if full_shape[d] == local_shape[d] * sp_size
        ]
        if len(shard_dims) != 1:
            raise ValueError(
                "Rollout with SP expects exactly one shard dimension "
                "(full_shape[d] == local_shape[d] * sp_world_size). "
                f"Got {len(shard_dims)} candidate dims (full_shape={full_shape}, "
                f"local_shape={tuple(local_shape)}, sp_world_size={sp_size}). "
                "Check that prepare_rollout(noise_full_shape=...) matches how "
                "latents are sharded (e.g. single time or sequence dim)."
            )
        shard_dim = shard_dims[0]
        device = model_output.device
        dtype = model_output.dtype
        buffer = self._get_or_create_rollout_noise_buffer(full_shape, device, dtype)
        torch.randn(*full_shape, out=buffer, generator=generator)
        rank = get_sp_parallel_rank()
        chunk_size = full_shape[shard_dim] // sp_size
        start = rank * chunk_size
        variance_noise = buffer.narrow(shard_dim, start, chunk_size)
        return variance_noise

    @require_rollout_enabled_decorator
    def flow_sde_sampling(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        current_sigma: torch.FloatTensor,
        next_sigma: torch.FloatTensor,
        generator: torch.Generator,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """flow sde sampling methods, reference: FlowGRPO"""

        variance_noise = self._rollout_variance_noise(model_output, generator)

        dt = next_sigma - current_sigma
        if self._rollout_param_sde_type == "sde":
            std_dev_t = torch.sqrt(
                current_sigma / 
                (1 - torch.where(torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                                               self._rollout_sigma_max, current_sigma))) * self._rollout_param_noise_level
            noise_std_dev = std_dev_t * torch.sqrt(-1*dt)
            prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt) \
                               + model_output * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma)) * dt

            prev_sample = prev_sample_mean + noise_std_dev * variance_noise
            log_prob_no_const = -((prev_sample - prev_sample_mean) ** 2)

        elif self._rollout_param_sde_type == "cps":
            std_dev_t = next_sigma * math.sin(self._rollout_param_noise_level * math.pi / 2) # sigma_t in paper
            noise_std_dev = std_dev_t # the std before noise in paper
            pred_original_sample = sample - current_sigma * model_output # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - current_sigma) # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

            prev_sample = prev_sample_mean + noise_std_dev * variance_noise
            log_prob_no_const = -((prev_sample - prev_sample_mean) ** 2)

        else:
            raise ValueError(f"Unsupported sde_type: {self._rollout_param_sde_type}")

        # Calculate local log_prob sum and local element count (for cross-SP mean).
        reduce_dims = list(range(1, len(log_prob_no_const.shape)))
        local_elem_count = log_prob_no_const.new_full(
            (log_prob_no_const.shape[0],),
            float(math.prod(log_prob_no_const.shape[1:])),
        )

        if self._rollout_param_log_prob_no_const:
            log_prob_local_sum = log_prob_no_const.sum(dim=reduce_dims)
        else:
            log_prob_local_sum = (
                log_prob_no_const / (2 * (noise_std_dev**2))
                - torch.log(noise_std_dev)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi).to(noise_std_dev.device)))
            ).sum(dim=list(range(1, len(log_prob_no_const.shape))))

        return prev_sample, log_prob_local_sum, local_elem_count

    # log prob utils for rollout
    @require_rollout_enabled_decorator
    def append_local_rollout_log_probs(
        self, log_prob_sum: torch.Tensor, log_prob_count: torch.Tensor
    ) -> None:
        self._rollout_local_log_prob_sum.append(log_prob_sum)
        self._rollout_local_log_prob_count.append(log_prob_count)

    @require_rollout_enabled_decorator
    def consume_local_rollout_log_probs(self) -> tuple[torch.Tensor, torch.Tensor]:
        values_sum = torch.stack(self._rollout_local_log_prob_sum, dim=-1)
        values_count = torch.stack(self._rollout_local_log_prob_count, dim=-1)
        self._rollout_local_log_prob_sum = []
        self._rollout_local_log_prob_count = []
        return values_sum, values_count

    @require_rollout_enabled_decorator
    def collect_rollout_log_probs(self, batch: Req) -> torch.Tensor | None:
        """Consume local rollout log probs and merge for all SP ranks."""

        trajectory_log_prob_sum, trajectory_log_prob_count = (
            self.consume_local_rollout_log_probs()
        )
        if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
            packed = torch.stack(
                [trajectory_log_prob_sum, trajectory_log_prob_count], dim=0
            ).to(
                get_local_torch_device()
            )
            sequence_model_parallel_all_reduce(packed)
            trajectory_log_prob_sum = packed[0]
            trajectory_log_prob_count = packed[1]

        # trajectory_log_probs_tensor: [B, T], reduced as global_mean = global_sum / global_count.
        trajectory_log_probs_tensor = trajectory_log_prob_sum / trajectory_log_prob_count
        return trajectory_log_probs_tensor.cpu()