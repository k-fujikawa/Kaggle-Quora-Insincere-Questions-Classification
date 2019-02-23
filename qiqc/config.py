import argparse
from abc import ABCMeta, abstractmethod
from pathlib import Path


class ExperimentConfigBuilderBase(metaclass=ABCMeta):

    default_config = None

    def add_args(self, parser):
        parser.add_argument('--modelfile', '-m', type=Path)
        parser.add_argument('--outdir-top', type=Path, default=Path('results'))
        parser.add_argument('--outdir-bottom', type=str, default='default')
        parser.add_argument('--device', '-g', type=int)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--logging', action='store_true')
        parser.add_argument('--n-rows', type=int)

        parser.add_argument('--seed', type=int, default=1029)
        parser.add_argument('--optuna-trials', type=int)
        parser.add_argument('--gridsearch', action='store_true')
        parser.add_argument('--holdout', action='store_true')
        parser.add_argument('--cv', type=int, default=5)
        parser.add_argument('--cv-part', type=int)
        parser.add_argument('--processes', type=int, default=2)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--batchsize', type=int, default=512)
        parser.add_argument('--batchsize-valid', type=int, default=1024)
        parser.add_argument('--scale-batchsize', type=int, nargs='+',
                            default=[])
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--validate-from', type=int)
        parser.add_argument('--pos-weight', type=float, default=1.)
        parser.add_argument('--maxlen', type=float, default=72)
        parser.add_argument('--vocab-mincount', type=float, default=5)
        parser.add_argument('--ensembler-n-snapshots', type=int, default=1)

    @abstractmethod
    def modules(self):
        raise NotImplementedError()

    def build(self, args=None):
        assert self.default_config is not None
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        parser.set_defaults(**self.default_config)

        for module in self.modules:
            module.add_args(parser)
        config, extra_config = parser.parse_known_args(args)

        for module in self.modules:
            if hasattr(module, 'add_extra_args'):
                module.add_extra_args(parser, config)

        if config.test:
            parser.set_defaults(**dict(
                n_rows=500,
                batchsize=64,
                validate_from=0,
                epochs=3,
                cv_part=2,
                ensembler_test_size=1.,
            ))

        config = parser.parse_args(args)
        if config.modelfile is not None:
            config.outdir = config.outdir_top / config.modelfile.stem \
                / config.outdir_bottom
        else:
            config.outdir = Path('.')

        return config
