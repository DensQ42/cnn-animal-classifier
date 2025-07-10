import numpy as np
import torch
import random

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across NumPy, Python's random, and PyTorch (CPU and GPU).

    This function ensures consistent results between runs by fixing the seed values
    for all major random number generators. It also enforces deterministic behavior
    in cuDNN-based operations, which may affect performance but ensures reproducibility.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.

    Notes:
        - `torch.backends.cudnn.deterministic = True` forces cuDNN to use deterministic algorithms.
        - `torch.backends.cudnn.benchmark = False` prevents cuDNN from benchmarking to select fastest kernels.
        - This setup may slow down training slightly but ensures reproducibility.
        - If using multiple worker processes (e.g., `num_workers > 0` in DataLoader), consider setting `worker_init_fn`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
