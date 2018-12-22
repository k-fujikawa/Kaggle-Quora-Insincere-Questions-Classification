import os
import random
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
import prompter


def rmtree_after_confirmation(path, force=False):
    if Path(path).exists():
        if not force and not prompter.yesno('Overwrite %s?' % path):
            sys.exit(0)
        else:
            shutil.rmtree(path)


def pad_sequence(xs, length, padding_value=0):
    assert isinstance(xs, list)
    n_padding = length - len(xs)
    return np.array(xs + [padding_value] * n_padding, 'i')[:length]


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
