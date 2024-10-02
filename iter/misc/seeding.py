import os
import random

import numpy as np
import torch


def setup_seed(seed: int, strict: bool = True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(min(seed, 2 ** 32 - 1))
    if strict:
        setup_reproducible()


def setup_reproducible():
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
