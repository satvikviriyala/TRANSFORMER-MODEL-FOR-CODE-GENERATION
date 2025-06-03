# src/utils/ddp_utils.py
import torch
import torch.distributed as dist
import os
from src.utils.logging_utils import setup_logger

logger = setup_logger("DDPUtils")

def setup_ddp():
    """Initializes the distributed process group."""
    backend = 'nccl' # Assumes NVIDIA GPUs
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    logger.info(f"DDP Setup: Rank {dist.get_rank()}/{dist.get_world_size()} on GPU {local_rank}")
    return local_rank, dist.get_world_size()

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
    logger.info("DDP Cleanup complete.")

def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def ddp_barrier():
    """Synchronizes all processes."""
    if dist.is_initialized():
        dist.barrier()