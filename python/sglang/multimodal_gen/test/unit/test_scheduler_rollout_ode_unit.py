import types
import unittest

import torch

import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as rl_mixin_module
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import SchedulerRLMixin


class _DummyScheduler(SchedulerRLMixin):
    def __init__(self):
        # Keep sigmas length >= 2 for _rollout_sigma_max initialization.
        self.sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=torch.float32)
        self.reset_rollout_states()


class TestSchedulerRolloutOdeUnit(unittest.TestCase):
    def setUp(self):
        self._orig_get_sp_world_size = rl_mixin_module.get_sp_world_size
        rl_mixin_module.get_sp_world_size = lambda: 1

    def tearDown(self):
        rl_mixin_module.get_sp_world_size = self._orig_get_sp_world_size

    def _build_batch(self, *, debug_mode: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            rollout_log_prob_no_const=True,
            rollout_noise_level=0.5,
            rollout_sde_type="ode",
            rollout_debug_mode=debug_mode,
            latents=torch.zeros(2, 4, 8, 8, dtype=torch.float32),
        )

    def test_ode_step_does_not_call_variance_noise_sampler(self):
        scheduler = _DummyScheduler()
        batch = self._build_batch(debug_mode=False)
        scheduler.prepare_rollout(batch)

        # ODE path is deterministic and must not invoke random variance sampling.
        def _raise_if_called(*args, **kwargs):
            raise AssertionError("ODE path should not call _rollout_variance_noise")

        scheduler._rollout_variance_noise = _raise_if_called  # type: ignore[method-assign]

        sample = torch.randn(2, 4, 8, 8, dtype=torch.float32)
        model_output = torch.randn_like(sample)
        current_sigma = torch.tensor(0.6, dtype=torch.float32)
        next_sigma = torch.tensor(0.4, dtype=torch.float32)

        prev_sample, log_prob_local_sum, local_elem_count = scheduler.flow_sde_sampling(
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=torch.Generator(device=sample.device).manual_seed(1),
        )

        expected_prev = sample + (next_sigma - current_sigma) * model_output
        self.assertTrue(torch.allclose(prev_sample, expected_prev, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(log_prob_local_sum, torch.zeros_like(log_prob_local_sum)))
        self.assertEqual(tuple(log_prob_local_sum.shape), (sample.shape[0],))
        self.assertEqual(tuple(local_elem_count.shape), (sample.shape[0],))
        self.assertTrue(torch.all(local_elem_count == float(sample[0].numel())))

    def test_ode_debug_tensors_have_shape_safe_noise_std(self):
        scheduler = _DummyScheduler()
        batch = self._build_batch(debug_mode=True)
        scheduler.prepare_rollout(batch)

        sample = torch.randn(2, 4, 8, 8, dtype=torch.float32)
        model_output = torch.randn_like(sample)
        current_sigma = torch.tensor(0.6, dtype=torch.float32)
        next_sigma = torch.tensor(0.4, dtype=torch.float32)

        scheduler.flow_sde_sampling(
            model_output=model_output,
            sample=sample,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            generator=torch.Generator(device=sample.device).manual_seed(2),
        )

        (
            variance_noises,
            prev_sample_means,
            noise_std_devs,
            model_outputs,
        ) = scheduler.consume_local_rollout_debug_tensors()

        # [B, T, ...] with one step in this test.
        self.assertEqual(tuple(variance_noises.shape), (2, 1, 4, 8, 8))
        self.assertEqual(tuple(prev_sample_means.shape), (2, 1, 4, 8, 8))
        self.assertEqual(tuple(model_outputs.shape), (2, 1, 4, 8, 8))
        self.assertEqual(tuple(noise_std_devs.shape), (2, 1, 1))
        self.assertTrue(torch.allclose(noise_std_devs, torch.zeros_like(noise_std_devs)))
        self.assertTrue(torch.allclose(variance_noises, torch.zeros_like(variance_noises)))


if __name__ == "__main__":
    unittest.main()

