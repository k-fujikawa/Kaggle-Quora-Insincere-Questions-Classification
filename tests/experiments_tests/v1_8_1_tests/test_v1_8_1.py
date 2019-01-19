import tempfile
import os
import shutil
from unittest import TestCase
from pathlib import Path

import pandas as pd
import yaml

import qiqc


class Test_v1_8_1(TestCase):

    modeldir = Path('experiments/bilstm/v1.8.1')

    def setUp(self):
        self.outdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.outdir, ignore_errors=True)

    def test_1epoch(self):
        topdir = Path(qiqc.__file__).parents[1]
        os.environ['DATADIR'] = str(topdir / 'tests/dummy_data')
        config = yaml.load(open(topdir / self.modeldir / 'config.yml'))
        config['outdir'] = self.outdir
        config['modeldir'] = self.modeldir
        config['n_rows'] = None
        config['batchsize'] = 8
        config['epochs'] = 1
        config['cv_part'] = 1
        config['test'] = True
        config['holdout'] = False
        config['device'] = None
        config['logging'] = True
        config['validate_from'] = 0
        config['ensembler']['test_size'] = 1

        mod = qiqc.loader.load_module(topdir / 'exec/train.py')
        mod.train(config)
        df_predicted = pd.read_csv(self.outdir / 'submission.csv')
        df_expected = pd.read_csv(Path(__file__).parent / 'submission.csv')
        self.assertTrue(
            (df_expected.prediction == df_predicted.prediction).all())
