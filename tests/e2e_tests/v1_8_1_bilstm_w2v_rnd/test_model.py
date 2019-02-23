import tempfile
import os
import shutil
from unittest import TestCase
from pathlib import Path

import pandas as pd

import qiqc


class Test_v1_8_1_BiLSTMRnd(TestCase):

    modelfile = Path('models/baseline/v1_8_1_bilstm_w2v_rnd.py')

    def setUp(self):
        self.outdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.outdir, ignore_errors=True)

    def test_1epoch(self):
        topdir = Path(qiqc.__file__).parents[1]
        os.environ['DATADIR'] = str(topdir / 'tests/dummy_data')
        args = f'''
        --outdir-top {self.outdir}
        --modelfile {self.modelfile}
        --batchsize 8
        --epoch 1
        --cv-part 1
        --test
        '''.split()

        mod = qiqc.utils.load_module(topdir / 'exec/train.py')
        mod.main(args=args)
        modelname = self.modelfile.stem
        df_predicted = pd.read_csv(
            self.outdir / f'{modelname}/default/submission.csv')
        df_expected = pd.read_csv(
            Path(__file__).parent / 'submission.csv')
        self.assertTrue(
            (df_expected.prediction == df_predicted.prediction).all())
