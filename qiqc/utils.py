import os
import random
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import prompter
from joblib import Parallel, delayed

from _qiqc.utils import *  # NOQA


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


def parallel_apply(df, f, axis=None, processes=2):
    dfs = np.array_split(df, processes)
    if axis is None:
        applyfunc = lambda x: x.apply(f)  # NOQA
    else:
        applyfunc = lambda x: x.apply(f, axis=axis)  # NOQA
    outputs = Parallel(n_jobs=processes)(delayed(applyfunc)(df) for df in dfs)
    return pd.concat(outputs)


class Pipeline(object):

    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x
