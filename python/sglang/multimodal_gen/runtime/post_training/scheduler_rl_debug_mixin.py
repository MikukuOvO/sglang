# SPDX-License-Identifier: Apache-2.0
"""Debug tensor helpers for rollout-enabled schedulers."""

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class SchedulerRLDebugMixin:
    def _reset_rollout_debug_tensors(self) -> None:
        self._rollout_local_variance_noises = []
        self._rollout_local_prev_sample_means = []
        self._rollout_local_noise_std_devs = []
        self._rollout_local_model_outputs = []

    def append_local_rollout_debug_tensors(
        self,
        *,
        variance_noise: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        noise_std_dev: torch.Tensor,
        model_output: torch.Tensor,
    ) -> None:
        self._require_rollout_enabled()
        batch_size = variance_noise.shape[0]
        self._rollout_local_variance_noises.append(variance_noise)
        self._rollout_local_prev_sample_means.append(prev_sample_mean)
        self._rollout_local_noise_std_devs.append(noise_std_dev.expand((batch_size, 1)))
        self._rollout_local_model_outputs.append(model_output)

    def consume_local_rollout_debug_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_rollout_enabled()
        variance_noises = torch.stack(self._rollout_local_variance_noises, dim=1)
        prev_sample_means = torch.stack(self._rollout_local_prev_sample_means, dim=1)
        noise_std_devs = torch.stack(self._rollout_local_noise_std_devs, dim=1)
        model_outputs = torch.stack(self._rollout_local_model_outputs, dim=1)
        self._reset_rollout_debug_tensors()
        return variance_noises, prev_sample_means, noise_std_devs, model_outputs

    def collect_rollout_debug_tensors(
        self, batch: Req
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Consume rollout debug tensors and merge for all SP ranks.

        Returns four tensors with shape [B, T, ...]:
        - variance_noises
        - prev_sample_means
        - noise_std_devs
        - model_outputs
        """
        self._require_rollout_enabled()
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
