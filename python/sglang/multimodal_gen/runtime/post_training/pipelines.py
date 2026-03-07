from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.post_training.models import RLSchedulerMixin
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_reduce,
)


class DenoisingRLMixin:
    def _maybe_prepare_rollout(self, batch: Req):
        """Prepare denoising loop for rollout"""
        if not isinstance(self.scheduler, RLSchedulerMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(self.scheduler)} does not support rollout"
                )
            return

        self.scheduler.reset_rollout_states()
        if batch.rollout:
            self.scheduler.prepare_rollout(
                noise_level=batch.rollout_noise_level,
                sde_type=batch.rollout_sde_type,
                log_prob_no_const=batch.rollout_log_prob_no_const,
            )
    
    def _maybe_get_rollout_log_probs(self, batch: Req):
        """Get log probs from rollout and add to server args for reward calculation"""
        if not isinstance(self.scheduler, RLSchedulerMixin):
             if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(self.scheduler)} does not support rollout"
                )
             return

        if batch.rollout:
            trajectory_log_probs_tensor = self.scheduler.consume_local_rollout_log_probs()
            if get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False):
                trajectory_tensor = trajectory_tensor.to(get_local_torch_device())
                # gather log probs across sequence parallel workers if using sequence parallelism
                sequence_model_parallel_all_reduce(trajectory_log_probs_tensor)
            batch.trajectory_log_probs = trajectory_log_probs_tensor.cpu()

    