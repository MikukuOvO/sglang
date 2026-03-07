# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

import math
from typing import Any, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


class SchedulerRLMixin:
    def reset_rollout_states(self):
        """Reset rollout states, should be called at the beginning of each new request"""
        self._rollout_enabled = False
        self._rollout_log_probs = []

    def prepare_rollout(
        self,
        *,
        noise_level: float = 0.7,
        sde_type: str = "sde",
        log_prob_no_const: bool = True
    ) -> None:
        self._rollout_enabled = True
        self._rollout_log_probs = []
        # Prepare params needed for rollout
        self._rollout_param_log_prob_no_const = log_prob_no_const
        self._rollout_param_noise_level = float(noise_level)
        self._rollout_param_sde_type = sde_type

        # Prepare extra parameters for sampling
        self._rollout_sigma_max = self.sigmas[min(1, len(self.sigmas) - 1)].item()
    
    def already_prepared_rollout(self):
        return getattr(self, "_rollout_enabled", False)

    def _require_rollout_enabled(self) -> None:
        if not getattr(self, "_rollout_enabled", False):
            raise RuntimeError("prepare_rollout() not called before rollout")

    def flow_sde_sampling(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        current_sigma: Union[float, torch.FloatTensor],
        next_sigma: Union[float, torch.FloatTensor],
        generator: torch.Generator,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """flow sde sampling methods, reference: FlowGRPO"""
        self._require_rollout_enabled()

        dt = next_sigma - current_sigma
        if self._rollout_param_sde_type == "sde":
            std_dev_t = torch.sqrt(
                current_sigma / 
                (1 - torch.where(torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                                               self._rollout_sigma_max, current_sigma))) * self._rollout_param_noise_level
            noise_std_dev = std_dev_t * torch.sqrt(-1*dt)
            prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt) \
                               + model_output * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma)) * dt
            
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + noise_std_dev * variance_noise
            log_prob_no_const = -((prev_sample - prev_sample_mean) ** 2)

        elif self._rollout_param_sde_type == "cps":
            std_dev_t = next_sigma * math.sin(self._rollout_param_noise_level * math.pi / 2) # sigma_t in paper
            noise_std_dev = std_dev_t # the std before noise in paper
            pred_original_sample = sample - current_sigma * model_output # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - current_sigma) # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + noise_std_dev * variance_noise
            log_prob_no_const = -((prev_sample - prev_sample_mean) ** 2)

        else:
            raise ValueError(f"Unsupported sde_type: {self._rollout_param_sde_type}")

        # Calculate log_prob
        if self._rollout_param_log_prob_no_const:
            log_prob = log_prob_no_const
        else :
            log_prob = (
                log_prob_no_const / (2 * (noise_std_dev**2))
                - torch.log(noise_std_dev)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        return prev_sample, log_prob

    # log prob utils for rollout
    def append_local_rollout_log_prob(self, log_prob: torch.Tensor) -> None:
        self._require_rollout_enabled()
        self._rollout_log_probs.append(log_prob)

    def consume_local_rollout_log_probs(self) -> torch.Tensor:
        self._require_rollout_enabled()
        values = torch.stack(self._rollout_log_probs, dim=-1)
        self._rollout_log_probs = []
        return values