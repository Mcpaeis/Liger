import random
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(random.randint(0, 1000))
    torch.manual_seed(random.randint(0, 1000))
    torch.cuda.manual_seed_all(random.randint(0, 1000))
    torch.backends.cudnn.deterministic = True
