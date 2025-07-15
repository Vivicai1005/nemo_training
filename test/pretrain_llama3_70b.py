import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.utils import logging

from typing import List
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from lightning.pytorch.callbacks.callback import Callback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback

def get_comm_overlap_callback_idx(callbacks: List[Callback]) -> int | None:
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                return idx
    return None


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_pretraining():
    recipe = llm.llama3_70b.pretrain_recipe(
        dir="/checkpoints/llama3-70b", # Path to store checkpoints
        name="llama3_pretraining",
        num_nodes=1,
        num_gpus_per_node=8)
    recipe.trainer.strategy.tensor_model_parallel_size=4
    recipe.trainer.strategy.pipeline_model_parallel_size=2
    recipe.trainer.strategy.context_parallel_size=1
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size=None
    recipe.data.global_batch_size=8
    recipe.data.micro_batch_size=1
    recipe.data.seq_length = 2048
    recipe.model.config.seq_length = 2048 # for flops calculation
    recipe.trainer.plugins = bf16_with_fp8_mixed()
    recipe.trainer.plugins.grad_reduce_in_fp32 = False
    # recipe.trainer.strategy.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    # recipe.optim.config.reuse_grad_buf_for_mxfp8_param_ag = True
    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    if comm_overlap_callback_idx is not None:
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather = False
    logging.warning(
        "When using MXFP8, to reduce memory usage, we use reuse_grad_buf_for_mxfp8_param_ag. "
        "Disabling AG overlap because it is not supported with reuse_grad_buf_for_mxfp8_param_ag."
    )

    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="llama3",
        )
    )

    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor, name="llama3_70b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()