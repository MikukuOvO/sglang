# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

import math
from functools import wraps
from typing import Any, Optional, Union

import torch
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
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
        self._rollout_ctx = None
        self._rollout_local_log_prob_sum = []
        self._rollout_local_log_prob_count = []
        self._rollout_local_variance_noises = []
        self._rollout_local_prev_sample_means = []
        self._rollout_local_noise_std_devs = []
        self._rollout_local_model_outputs = []

    def reset_rollout_states(self):
        """Reset rollout states, should be called at the beginning of each new request"""
        self._rollout_enabled = False
        self.release_rollout_resources()

    def prepare_rollout(self, batch: Req) -> None:
        """Enable rollout and set SDE/CPS params. Call once before the denoising loop."""
        self._rollout_enabled = True
        self._rollout_local_log_prob_sum = []
        self._rollout_local_log_prob_count = []
        self._rollout_local_variance_noises = []
        self._rollout_local_prev_sample_means = []
        self._rollout_local_noise_std_devs = []
        self._rollout_local_model_outputs = []
        log_prob_no_const = batch.rollout_log_prob_no_const
        pipeline_config = getattr(batch, "_rollout_pipeline_config", None)
        if get_sp_world_size() > 1 and pipeline_config is None:
            raise RuntimeError(
                "SP rollout requires batch._rollout_pipeline_config to be set before prepare_rollout()."
            )
        # Prepare params needed for rollout
        self._rollout_param_log_prob_no_const = log_prob_no_const
        self._rollout_param_noise_level = float(batch.rollout_noise_level)
        self._rollout_param_sde_type = batch.rollout_sde_type
        self._rollout_latents_shape = tuple(batch.latents.shape) if batch.latents is not None else None
        # Use rollout_ctx to store any external context needed for rollout
        self._rollout_ctx = {
            "pipeline_config": pipeline_config,
            "batch": batch
        }

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
        """Get or create the reusable noise buffer (local or full shape) for rollout."""
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
        generator: Union[torch.Generator, list[torch.Generator]],
    ) -> torch.FloatTensor:
        """Generate variance noise for rollout. If generator is a list, use generator[i] for the i-th batch item."""
        assert generator is not None, "Generator must be provided"

        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)
        batch = self._rollout_ctx["batch"]
        pipeline_config = self._rollout_ctx["pipeline_config"]

        # Check generator validity
        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            assert B == 1, "Generator must be a list if batch size is not 1"
            generator = [generator]
        else:
            assert len(generator) == B, "Generator list must have the same length as batch size"

        buffer = self._get_or_create_rollout_noise_buffer(self._rollout_latents_shape, device, dtype)
        for i in range(B):
            torch.randn(self._rollout_latents_shape, out=buffer[i : i + 1], generator=generator[i])

        sharded_noise, _ = pipeline_config.shard_latents_for_sp(batch, buffer)
        if tuple(sharded_noise.shape) != local_shape:
            raise ValueError(
                "Rollout SP noise shape mismatch after shard. "
                f"Expected local_shape={local_shape}, got {tuple(sharded_noise.shape)}."
            )
        return sharded_noise

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

        self.append_local_rollout_debug_tensors(
            variance_noise=variance_noise,
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
            model_output=model_output,
        )

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

    @require_rollout_enabled_decorator
    def append_local_rollout_debug_tensors(
        self,
        *,
        variance_noise: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        noise_std_dev: torch.Tensor,
        model_output: torch.Tensor,
    ) -> None:
        batch_size = variance_noise.shape[0]
        self._rollout_local_variance_noises.append(variance_noise)
        self._rollout_local_prev_sample_means.append(prev_sample_mean)
        self._rollout_local_noise_std_devs.append(noise_std_dev.expand((batch_size, 1)))
        self._rollout_local_model_outputs.append(model_output)

    @require_rollout_enabled_decorator
    def consume_local_rollout_debug_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        variance_noises = torch.stack(self._rollout_local_variance_noises, dim=1)
        prev_sample_means = torch.stack(self._rollout_local_prev_sample_means, dim=1)
        noise_std_devs = torch.stack(self._rollout_local_noise_std_devs, dim=1)
        model_outputs = torch.stack(self._rollout_local_model_outputs, dim=1)
        self._rollout_local_variance_noises = []
        self._rollout_local_prev_sample_means = []
        self._rollout_local_noise_std_devs = []
        self._rollout_local_model_outputs = []
        return variance_noises, prev_sample_means, noise_std_devs, model_outputs

    @require_rollout_enabled_decorator
    def collect_rollout_debug_tensors(
        self, batch: Req
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Consume rollout debug tensors and merge for all SP ranks.

        Returns three tensors with shape [B, T, ...]:
        - variance_noises
        - prev_sample_means
        - noise_std_devs
        - model_outputs
        """
        variance_noises, prev_sample_means, noise_std_devs, model_outputs = (
            self.consume_local_rollout_debug_tensors()
        )

        if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
            variance_noises = variance_noises.to(get_local_torch_device())
            prev_sample_means = prev_sample_means.to(get_local_torch_device())
            noise_std_devs = noise_std_devs.to(get_local_torch_device())
            model_outputs = model_outputs.to(get_local_torch_device())
            pipeline_config = self._rollout_ctx["pipeline_config"]
            bsz, num_steps = variance_noises.shape[0], variance_noises.shape[1]

            # [B, T, ...] -> [B*T, ...]
            variance_noises_packed = variance_noises.contiguous().reshape(
                bsz * num_steps, *variance_noises.shape[2:]
            )
            prev_sample_means_packed = prev_sample_means.contiguous().reshape(
                bsz * num_steps, *prev_sample_means.shape[2:]
            )
            model_outputs_packed = model_outputs.contiguous().reshape(
                bsz * num_steps, *model_outputs.shape[2:]
            )

            # Gather on packed tensors first.
            variance_noises_packed = pipeline_config.gather_latents_for_sp(
                variance_noises_packed
            )
            prev_sample_means_packed = pipeline_config.gather_latents_for_sp(
                prev_sample_means_packed
            )
            model_outputs_packed = pipeline_config.gather_latents_for_sp(
                model_outputs_packed
            )

            # Unpack back to [B, T, ...].
            variance_noises = variance_noises_packed.reshape(
                bsz, num_steps, *variance_noises_packed.shape[1:]
            )
            prev_sample_means = prev_sample_means_packed.reshape(
                bsz, num_steps, *prev_sample_means_packed.shape[1:]
            )
            model_outputs = model_outputs_packed.reshape(
                bsz, num_steps, *model_outputs_packed.shape[1:]
            )
            # noise_std_devs is same on every device, not a sharded latent tensor.

        return (
            variance_noises.cpu(),
            prev_sample_means.cpu(),
            noise_std_devs.cpu(),
            model_outputs.cpu(),
        )
