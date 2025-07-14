import nemo_run as run

def local_executor_torchrun(gpu: str,
                            nodes: int,
                            num_gpus_per_node: int,
                            hf_token: str = None) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    PERF_ENV_VARS = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
    }

    if gpu.lower() not in ['b200']:
        # TODO: we currently disable PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        # on B200 as it causes an unexpected error. Add back when issue is debugged and fixed.
        PERF_ENV_VARS["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if hf_token is not None:
        PERF_ENV_VARS.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})


    executor = run.LocalExecutor(
                                 ntasks_per_node=num_gpus_per_node,
                                 launcher="torchrun",
                                 env_vars=PERF_ENV_VARS,
                                 )

    return executor