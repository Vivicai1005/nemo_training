import nemo_run as run

from typing import List
from nemo.collections import llm
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from lightning.pytorch.callbacks.callback import Callback
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.utils import logging


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
    recipe = llm.llama3_8b.pretrain_recipe(
        dir="/checkpoints/llama3", # Path to store checkpoints
        name="llama3_pretraining",
        num_nodes=1,
        num_gpus_per_node=8)

    recipe.trainer.strategy.tensor_model_parallel_size = 1
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    recipe.data.global_batch_size = 128
    recipe.data.micro_batch_size = 2
    recipe.data.seq_length = 8192
    recipe.trainer.plugins = bf16_with_fp8_mixed()
    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="llama3",
        )
    )

    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor, name="llama3_8b_pretraining")

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()