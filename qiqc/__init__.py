import os
import random

import numpy as np
import torch

import qiqc.builder  # NOQA
import qiqc.config  # NOQA
import qiqc.loader  # NOQA
import qiqc.models  # NOQA
import qiqc.model_selection  # NOQA
import qiqc.preprocessors  # NOQA
import qiqc.utils  # NOQA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
