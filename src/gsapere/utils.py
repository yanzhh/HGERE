import logging
import os
import random
import sys

import numpy as np
import torch


def get_logger(args, log_path, test: bool = False):
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)
    if test:
        log_file = f"test_{args.hostname}.log"
    else:
        log_file = f"all_{args.hostname}.log"
    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    log_level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    logger.setLevel(log_level)

    # not used: datefmt="%m/%d/%Y %H:%M:%S",
    return logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for deterministic cuBLAS (linear layers, attention matmul).
    # Must be set before any CUDA kernel launches; setting the env var here
    # is safe because PyTorch reads it lazily on first cuBLAS workspace alloc.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
