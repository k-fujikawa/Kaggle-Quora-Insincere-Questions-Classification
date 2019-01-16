import tempfile
import shutil
from unittest import TestCase
from pathlib import Path

import numpy as np
import yaml

import qiqc


class Test_v1_8_1(TestCase):

    modeldir = 'experiments/bilstm/v1.8.1'
    valid_fbeta = 0.13157894736842105

    def setUp(self):
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.outdir, ignore_errors=True)

    def test_1epoch(self):
        topdir = Path(qiqc.__file__).parents[1]
        config = yaml.load(open(topdir / self.modeldir / 'config.yml'))
        config['outdir'] = Path(self.outdir)
        config['modeldir'] = Path(self.modeldir)
        config['n_rows'] = 1000
        config['epochs'] = 1
        config['cv_part'] = 1
        config['test'] = True
        config['holdout'] = False
        config['device'] = None
        config['validate_from'] = 0

        mod = qiqc.loader.load_module(topdir / 'exec/train.py')
        result = mod.train(config)
        np.testing.assert_allclose(result['valid_fbeta'], self.valid_fbeta)
